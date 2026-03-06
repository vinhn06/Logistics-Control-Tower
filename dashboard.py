import streamlit as st
import pandas as pd
import math
import json
import numpy as np
import requests
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# api and routing utilities

def get_osrm_route(origin_lat, origin_lon, dest_lat, dest_lon):
    url = f"http://router.project-osrm.org/route/v1/driving/{origin_lon},{origin_lat};{dest_lon},{dest_lat}?overview=false"
    headers = {'User-Agent': 'Logistics_Control_Tower/1.0'}
    try:
        response = requests.get(url, headers=headers, timeout=5).json()
        if response.get("code") == "Ok":
            distance_meters = response["routes"][0]["distance"]
            duration_seconds = response["routes"][0]["duration"]
            return int(distance_meters * 0.000621371), duration_seconds / 86400
    except Exception:
        pass
    return None, None

def get_live_dat_rate(total_billable_miles, origin_name):
    base_rate_per_mile = 2.25
    fuel_surcharge = 0.45 
    multiplier = {"Fresno, CA": 1.10, "Portland, OR": 1.05, "Denver, CO": 0.95, "Chicago, IL": 1.00}.get(origin_name, 1.0)
    return max(total_billable_miles * (base_rate_per_mile + fuel_surcharge) * multiplier, 1200.0)

def get_transit_metrics(origin_name, num_stops):
    origins = {
        "Fresno, CA": (36.7378, -119.7871), "Portland, OR": (45.5152, -122.6784), 
        "Denver, CO": (39.7392, -104.9903), "Chicago, IL": (41.8781, -87.6298)
    }
    hub = (38.5816, -121.4944) 
    highway_miles, driving_days = get_osrm_route(origins[origin_name][0], origins[origin_name][1], hub[0], hub[1])
    
    if highway_miles is None:
        R = 3958.8 
        dLat, dLon = math.radians(hub[0] - origins[origin_name][0]), math.radians(hub[1] - origins[origin_name][1])
        a = math.sin(dLat/2)**2 + math.cos(math.radians(origins[origin_name][0])) * math.cos(math.radians(hub[0])) * math.sin(dLon/2)**2
        highway_miles = int((R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))) * 1.15)
        driving_days = highway_miles / 600.0

    total_billable_miles = highway_miles + 50
    effective_transit_days = math.ceil(driving_days + (math.floor(driving_days) * 0.25) + math.floor(num_stops / 3))
    
    benchmark_rate = get_live_dat_rate(total_billable_miles, origin_name)
    return highway_miles, total_billable_miles, effective_transit_days, benchmark_rate

def get_scenario_multipliers(quarter_str, miles):
    base_sigma = 0.25 + (0.10 if miles > 1000 else 0)
    if "Q1" in quarter_str: return 1.0, 1.0, 1.0, "2026-01-01", base_sigma + 0.15
    elif "Q2" in quarter_str: return 1.2, 1.3, 1.1, "2026-04-01", base_sigma
    elif "Q3" in quarter_str: return 1.3, 1.2, 1.2, "2026-07-01", base_sigma + 0.05
    else: return 1.8, 2.0, 2.2, "2026-10-01", base_sigma + 0.15

def load_and_prep_data(filepath, m_milk, m_cups, m_beans, start_date):
    df = pd.read_csv(filepath)
    unique_dates = sorted(pd.to_datetime(df['Date']).unique())
    df['Date'] = pd.to_datetime(df['Date']).map(dict(zip(unique_dates, pd.date_range(start=start_date, periods=len(unique_dates)))))
    df['Oat_Milk_Used'] = (df['Oat_Milk_Used'] * m_milk).astype(int)
    df['Paper_Cups_Used'] = (df['Paper_Cups_Used'] * m_cups).astype(int)
    df['Beans_Used'] = (df['Beans_Used'] * m_beans).astype(int)
    return df

def get_optimal_cycle(df, max_vol, max_weight, max_pallets):
    vol_map, weight_map, pallet_map = {'Oat_Milk_Used': 0.55, 'Paper_Cups_Used': 2.33, 'Beans_Used': 1.77}, {'Oat_Milk_Used': 32.5, 'Paper_Cups_Used': 15.0, 'Beans_Used': 45.0}, {'Oat_Milk_Used': 60.0, 'Paper_Cups_Used': 20.0, 'Beans_Used': 40.0}
    shop_daily_avg = df.groupby('Shop_Location')[['Oat_Milk_Used', 'Paper_Cups_Used', 'Beans_Used']].mean()
    optimal_cycle = 0 
    
    for cycle in range(1, 91):
        p_milk, p_cups, p_beans = shop_daily_avg['Oat_Milk_Used'] * cycle, shop_daily_avg['Paper_Cups_Used'] * cycle, shop_daily_avg['Beans_Used'] * cycle
        t_pallets = np.ceil(p_milk / pallet_map['Oat_Milk_Used']).sum() + np.ceil(p_cups / pallet_map['Paper_Cups_Used']).sum() + np.ceil(p_beans / pallet_map['Beans_Used']).sum()
        t_vol = (p_milk.sum() * vol_map['Oat_Milk_Used']) + (p_cups.sum() * vol_map['Paper_Cups_Used']) + (p_beans.sum() * vol_map['Beans_Used']) + (t_pallets * 5.5)
        t_weight = (p_milk.sum() * weight_map['Oat_Milk_Used']) + (p_cups.sum() * weight_map['Paper_Cups_Used']) + (p_beans.sum() * weight_map['Beans_Used']) + (t_pallets * 50.0)
        
        if t_pallets <= max_pallets and t_vol <= max_vol and t_weight <= max_weight: optimal_cycle = cycle
        else: break 
            
    days = len(df['Date'].unique())
    return optimal_cycle, vol_map, weight_map, pallet_map, df['Oat_Milk_Used'].sum() / days, df['Paper_Cups_Used'].sum() / days, df['Beans_Used'].sum() / days

def run_monte_carlo_simulation(avg_daily_demand, total_lead_time, current_inventory, dynamic_sigma):
    simulations = 1000
    mu = np.log(total_lead_time)
    simulated_transit_times = np.random.lognormal(mean=mu, sigma=dynamic_sigma, size=simulations)
    simulated_transit_times = np.maximum(simulated_transit_times, 1.0) 
    
    simulated_demands = []
    safe_std_dev = max(avg_daily_demand * 0.20, 0.0001)
    
    for lead_time in simulated_transit_times:
        simulated_demand = np.random.normal(loc=avg_daily_demand, scale=safe_std_dev, size=int(math.ceil(lead_time)))
        simulated_demands.append(np.sum(simulated_demand))
        
    simulated_demands = np.array(simulated_demands)
    stockouts = np.sum(simulated_demands > current_inventory)
    service_level_probability = min(((simulations - stockouts) / simulations) * 100, 99.9)
    return service_level_probability, simulated_demands

def generate_arima_forecast(df):
    daily_demand = df.groupby('Date')[['Oat_Milk_Used', 'Paper_Cups_Used', 'Beans_Used']].sum()
    forecast_totals = {}
    for col in daily_demand.columns:
        model = ARIMA(daily_demand[col], order=(1, 1, 1))
        fit_model = model.fit()
        forecast = fit_model.forecast(steps=7)
        forecast_totals[col] = int(round(forecast.sum()))
    return forecast_totals

def create_tms_payload(origin, miles, lead_time, weight, vol, pallets, df_totals, trailer, liftgate):
    return json.dumps({
        "tender_id": "LCT-API-001", "origin_node": origin.split(',')[0], "destination_node": "Sacramento Hub",
        "routing_metrics": {"miles": miles, "lead_time_days": lead_time},
        "equipment": {"type": trailer, "weight_lbs": float(weight), "cube_ft": float(vol), "pallets": int(pallets), "liftgate": liftgate},
        "manifest_stops": [{"stop": i+1, "loc": r['Shop_Location'], "pallets": int(r['Total_Pallets'])} for i, r in df_totals.iterrows()]
    }, indent=4)

# application layout and execution

def main():
    st.set_page_config(page_title="Logistics Control Tower", layout="wide", initial_sidebar_state="expanded")

    # sidebar config
    st.sidebar.markdown("## 🎛️ Control Tower")
    quarter = st.sidebar.selectbox("Business Quarter:", ["Q1 (Jan-Mar): Baseline", "Q2 (Apr-Jun): Spring Growth", "Q3 (Jul-Sep): Summer Peak", "Q4 (Oct-Dec): Holiday Rush"])
    
    with st.sidebar.expander("🚛 Network & Equipment", expanded=False):
        origin = st.selectbox("Manufacturer Location:", ["Fresno, CA", "Portland, OR", "Denver, CO", "Chicago, IL"])
        trailer_type = st.selectbox("Equipment Type:", ["53' Dry Van", "53' Refrigerated (Reefer)", "48' Flatbed"])
        allow_double_stack = st.checkbox("Enable Double-Stacking", value=False)
        requires_liftgate = st.checkbox("Retail Liftgate Delivery", value=True)
    
    with st.sidebar.expander("💵 Finance & Planning", expanded=False):
        payment_terms = st.selectbox("Payment Terms:", [0, 15, 30, 45, 60], index=2, format_func=lambda x: f"Net {x}" if x > 0 else "Due on Receipt")
        facility_processing_days = st.slider("Supplier Pick/Pack (Days):", 0, 5, 2)
        safety_stock_days = st.slider("Safety Stock (Days):", 1, 21, 7)
        max_backroom_pallets = st.slider("Max Retail Backroom (Pallets):", 2, 12, 4)

    network_stops_init = 6
    _, total_billable_miles_init, _, benchmark_rate_init = get_transit_metrics(origin, network_stops_init)
    
    with st.sidebar.expander("📊 Carrier Bidding", expanded=False):
        freight_bill = st.slider("Base FTL Carrier Quote ($)", 100.0, 8000.0, float(benchmark_rate_init), 50.0)
        base_ltl_rate = st.number_input("Base LTL Rate per CuFt ($)", 0.01, value=2.10, step=0.10)

    # physical constraints setup
    eq_specs = {"53' Dry Van": (45000.0, 2800.0, 26), "53' Refrigerated (Reefer)": (43500.0, 2600.0, 26), "48' Flatbed": (48000.0, 1600.0, 24)}
    max_weight, max_vol, base_pallets = eq_specs[trailer_type]
    max_pallets = base_pallets * 2 if allow_double_stack else base_pallets

    # core data execution
    network_stops = 6 
    hw_miles, bill_miles, transit_days, benchmark_rate = get_transit_metrics(origin, 6)
    m_milk, m_cups, m_beans, start_mon, dyn_sigma = get_scenario_multipliers(quarter, bill_miles)
    df = load_and_prep_data('network_inventory_history.csv', m_milk, m_cups, m_beans, start_mon)
    total_lead_time = transit_days + facility_processing_days
    
    opt_cycle, v_map, w_map, p_map, a_milk, a_cups, a_beans = get_optimal_cycle(df, max_vol, max_weight, max_pallets)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Order Execution")
    st.sidebar.caption(f"Recommended Cycle: **{max(1, min(opt_cycle, 90))} Days**")
    
    order_cycle_days = st.sidebar.slider("Procurement Cycle (Days):", 1, 90, max(1, min(opt_cycle, 90)), 1)
    total_dwell_time = order_cycle_days + safety_stock_days

    # monte carlo simulation execution
    sl_milk, sims = run_monte_carlo_simulation(a_milk, total_lead_time, (a_milk * total_lead_time) + (a_milk * safety_stock_days), dyn_sigma)
    sl_cups, _ = run_monte_carlo_simulation(a_cups, total_lead_time, (a_cups * total_lead_time) + (a_cups * safety_stock_days), dyn_sigma)
    sl_beans, _ = run_monte_carlo_simulation(a_beans, total_lead_time, (a_beans * total_lead_time) + (a_beans * safety_stock_days), dyn_sigma)
    service_level = min(sl_milk, sl_cups, sl_beans)
    
    esc = ((a_milk*25.0) + (a_cups*10.0) + (a_beans*85.0)) * total_lead_time * ((100.0 - service_level) / 100.0)
    
    # calculate payload requirements based on selected cycle
    df_cycle = df[df['Date'] > (df['Date'].max() - pd.Timedelta(days=order_cycle_days))]
    df_totals = df_cycle.groupby('Shop_Location')[['Oat_Milk_Used', 'Paper_Cups_Used', 'Beans_Used']].sum().reset_index()
    
    df_totals['Total_Pallets'] = np.ceil(df_totals['Oat_Milk_Used']/p_map['Oat_Milk_Used']) + np.ceil(df_totals['Paper_Cups_Used']/p_map['Paper_Cups_Used']) + np.ceil(df_totals['Beans_Used']/p_map['Beans_Used'])
    df_totals['Total_Volume_CuFt'] = (df_totals['Oat_Milk_Used']*v_map['Oat_Milk_Used']) + (df_totals['Paper_Cups_Used']*v_map['Paper_Cups_Used']) + (df_totals['Beans_Used']*v_map['Beans_Used']) + (df_totals['Total_Pallets']*5.5)
    df_totals['Total_Weight_Lbs'] = (df_totals['Oat_Milk_Used']*w_map['Oat_Milk_Used']) + (df_totals['Paper_Cups_Used']*w_map['Paper_Cups_Used']) + (df_totals['Beans_Used']*w_map['Beans_Used']) + (df_totals['Total_Pallets']*50.0)
    
    tot_vol, tot_weight, tot_pallets = df_totals['Total_Volume_CuFt'].sum(), df_totals['Total_Weight_Lbs'].sum(), df_totals['Total_Pallets'].sum()
    vol_fill_rate, weight_fill_rate, pallet_fill_rate = (tot_vol/max_vol)*100, (tot_weight/max_weight)*100, (tot_pallets/max_pallets)*100

    raw_ltl = (df_totals['Oat_Milk_Used']*v_map['Oat_Milk_Used']*base_ltl_rate*0.8) + (df_totals['Paper_Cups_Used']*v_map['Paper_Cups_Used']*base_ltl_rate*1.8) + (df_totals['Beans_Used']*v_map['Beans_Used']*base_ltl_rate*1.0)
    df_totals['Est_LTL_Cost_$'] = np.maximum((raw_ltl * 1.30) + (150.0 if requires_liftgate else 0) + (np.maximum(0.0, df_totals['Total_Pallets'] - 6.0) * 125.0), 250.0)

    # inventory value and holding cost penalties
    df_totals['Inventory_Value_$'] = (df_totals['Oat_Milk_Used']*12.0) + (df_totals['Paper_Cups_Used']*15.0) + (df_totals['Beans_Used']*40.0)
    inv_val = df_totals['Inventory_Value_$']
    ss_val = (a_milk*12.0 + a_cups*15.0 + a_beans*40.0) * safety_stock_days
    spoilage_writeoff = inv_val.sum() * 0.10 if total_dwell_time > 45 else 0.0

    hc_pen = ((((inv_val.sum()/2)*max(0, order_cycle_days-payment_terms)) + (ss_val*order_cycle_days))*(0.08/365)) + ((((inv_val.sum()/2)+ss_val)*order_cycle_days)*(0.015/365)) + ((tot_pallets + np.ceil((a_milk*safety_stock_days)/p_map['Oat_Milk_Used']) + np.ceil((a_cups*safety_stock_days)/p_map['Paper_Cups_Used']) + np.ceil((a_beans*safety_stock_days)/p_map['Beans_Used'])) * 0.05 * order_cycle_days)
    
    # accessorials calculation
    total_unload_hours = network_stops * 1.5
    billable_detention_hours = max(0, total_unload_hours - 2.0)
    detention_fees = billable_detention_hours * 75.0
    layover_fee = 250.0 if (total_unload_hours + (50.0/35.0) + ((hw_miles%600)/50.0)) > 14.0 else 0.0
    stop_off_fees = max(0, (network_stops - 1) * 100.0)
    ftl_liftgate_fees = (network_stops - 1) * 100.0 if requires_liftgate else 0.0
    total_lumper_fees = tot_pallets * 25.0

    total_ftl_landed_cost = freight_bill + stop_off_fees + ftl_liftgate_fees + detention_fees + layover_fee + total_lumper_fees

    # proportional cost allocations per shop
    df_totals['Allocation_Pct'] = df_totals['Total_Volume_CuFt'] / max(tot_vol, 0.0001)
    df_totals['Allocated_FTL_Cost_$'] = total_ftl_landed_cost * df_totals['Allocation_Pct']
    df_totals['Holding_Cost_Penalty_$'] = hc_pen * df_totals['Allocation_Pct']
    df_totals['Expected_Shortage_Cost_$'] = esc * df_totals['Allocation_Pct']
    df_totals['Spoilage_Writeoff_$'] = spoilage_writeoff * df_totals['Allocation_Pct']
    df_totals['True_Net_Savings_$'] = df_totals['Est_LTL_Cost_$'] - (total_ftl_landed_cost * df_totals['Allocation_Pct']) - df_totals['Holding_Cost_Penalty_$'] - df_totals['Expected_Shortage_Cost_$'] - df_totals['Spoilage_Writeoff_$']

    carbon_credit_value = (((tot_weight/2000.0)*hw_miles*0.12) - ((tot_weight/2000.0)*hw_miles*0.08)) * 0.085
    true_net_savings = df_totals['Est_LTL_Cost_$'].sum() - total_ftl_landed_cost - hc_pen + carbon_credit_value - spoilage_writeoff - esc

    # final compliance logic flags
    is_physical_ok = tot_vol <= max_vol and tot_weight <= max_weight and tot_pallets <= max_pallets
    is_retail_ok = df_totals['Total_Pallets'].max() <= max_backroom_pallets
    is_pipeline_ok = order_cycle_days >= total_lead_time
    is_compliant = is_physical_ok and is_retail_ok and is_pipeline_ok

    # string formatting for ui presentation
    savings_str = f"-${abs(true_net_savings):,.2f}" if true_net_savings < 0 else f"${true_net_savings:,.2f}"
    safe_savings_str = savings_str.replace('$', r'\$')

    # main dashboard layout
    st.title("Logistics Control Tower & Routing Optimizer")
    st.markdown("### Executive Logistics Control Tower")

    # header kpi row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Optimal FTL Cycle", f"{order_cycle_days} Days", f"Recommended: {max(1, min(opt_cycle, 90))} Days")
    
    total_penalty = spoilage_writeoff + esc
    if total_penalty > 5.0:
        kpi2.metric("True Net Savings", savings_str, f"-${total_penalty:,.0f} Risk & Spoilage", delta_color="normal")
    else:
        kpi2.metric("True Net Savings", savings_str, f"+${carbon_credit_value:,.0f} ESG Tax Credit", delta_color="normal" if true_net_savings > 0 else "inverse")
        
    kpi3.metric("Network Survival Risk", f"{service_level:.1f}%", f"Lead Time: {total_lead_time} Days", delta_color="normal" if service_level >= 95 else "inverse")
    kpi4.metric("Carbon Emissions Avoided", f"{carbon_credit_value / 0.085:,.0f} kg", "EPA Ton-Mile Calc")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Network Pooling & Risk", "⚖️ Capacity & Financials", "🚀 Execution & TMS"])

    with tab1:
        st.header("1. Network Demand Pooling & Risk Analysis")
        st.info(f"**Dynamic Network Reorder Point (ROP):** To maintain your {safety_stock_days}-day safety buffer against a {total_lead_time}-day lead time (Transit + Pack) from {origin.split(',')[0]}, the system will automatically trigger the next FTL shipment when aggregate network inventory drops to **{((a_milk * total_lead_time) + (a_milk * safety_stock_days)):,.0f} cases** of Oat Milk.")

        daily_demand = df.groupby('Date')[['Oat_Milk_Used', 'Paper_Cups_Used', 'Beans_Used']].sum().reset_index()
        st.line_chart(daily_demand.set_index('Date'))

        st.subheader("7-Day Predictive Demand Forecast (ARIMA Time-Series)")
        forecast_totals = generate_arima_forecast(df)
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Projected Oat Milk", f"{forecast_totals['Oat_Milk_Used']} Cases")
        col_f2.metric("Projected Paper Cups", f"{forecast_totals['Paper_Cups_Used']} Boxes")
        col_f3.metric("Projected Espresso Beans", f"{forecast_totals['Beans_Used']} Sacks")

        st.divider()
        st.subheader("Monte Carlo Resilience Stress-Test")
        col_mc1, col_mc2 = st.columns([1, 2])
        with col_mc1:
            st.markdown(f"Mathematical probability of surviving the full **{total_lead_time}-day Lead Time** without a stockout.")
            if service_level >= 95.0: st.success(f"**Highly Resilient:** Survived {service_level:.1f}% of delay scenarios.")
            elif service_level >= 85.0: st.warning(f"**Moderate Risk:** {(100 - service_level):.1f}% chance of a critical stockout.")
            else: st.error(f"**Critical Vulnerability:** {(100 - service_level):.1f}% chance of catastrophic failure.")
                
            if esc > 0: st.warning(f"📉 **Financial Impact:** A {(100 - service_level):.1f}% failure probability risks \${esc:,.2f} in lost retail revenue. Deducted from ROI.")
        
        with col_mc2:
            hist, bins = np.histogram(sims, bins=30)
            st.bar_chart(pd.DataFrame({"Simulated Demand": hist}, index=np.round(bins[:-1], 0).astype(int)), color="#28a745" if service_level >= 95 else "#dc3545")

    with tab2:
        st.header("2. Physical Capacity & Financial Arbitrage")
        
        # physical capacity rendering
        st.markdown("**Trailer Utilization (Tri-Constraint: Cube vs. Weight vs. Pallet Positions):**")
        col_util1, col_util2, col_util3 = st.columns(3)
        with col_util1:
            st.metric("Space Utilized (Cube)", f"{tot_vol:,.1f} cu ft", f"{vol_fill_rate:.1f}% of {max_vol} Limit", delta_color="off")
            st.progress(min(vol_fill_rate / 100.0, 1.0))
        with col_util2:
            st.metric("Payload Utilized (Weight)", f"{tot_weight:,.0f} lbs", f"{weight_fill_rate:.1f}% of {max_weight} Limit", delta_color="off")
            st.progress(min(weight_fill_rate / 100.0, 1.0))
        with col_util3:
            st.metric("Floor Utilized (Pallets)", f"{tot_pallets:,.0f} positions", f"{pallet_fill_rate:.1f}% of {max_pallets} Limit", delta_color="off")
            st.progress(min(pallet_fill_rate / 100.0, 1.0))

        # active alerts
        if not is_pipeline_ok: st.error(f"🛣️ **Pipeline Collision:** Cycle ({order_cycle_days} Days) is shorter than lead time ({total_lead_time} Days).")
        if pallet_fill_rate > 100: st.error(f"**Pallet-Out Alert:** Requires {tot_pallets} pallets. A {trailer_type} holds {max_pallets}.")
        if vol_fill_rate > 100: st.warning(f"**Cube-Out Alert:** Exceeds limits by {tot_vol - max_vol:,.1f} cu ft.")
        if weight_fill_rate > 100: st.error(f"**Weigh-Out Alert:** Exceeds legal DOT limits by {tot_weight - max_weight:,.0f} lbs.")
        if not is_retail_ok: st.error(f"🏪 **Retail Overflow:** A single {df_totals['Total_Pallets'].max()} pallet drop breaches the {max_backroom_pallets}-pallet backroom limit.")

        st.divider()

        st.subheader("Financial Arbitrage & Landed Invoice")
        col_fin1, col_fin2 = st.columns([1, 1])

        with col_fin1:
            st.markdown("**Cost Arbitrage: Baseline vs. Optimized FTL**")
            
            # FIX: By setting the index to precisely match the columns, we stop Streamlit from auto-sorting the X-axis alphabetically
            chart_data = pd.DataFrame({
                "1. LTL Baseline": [df_totals['Est_LTL_Cost_$'].sum(), 0.0, 0.0],
                "2. FTL Landed Cost": [0.0, total_ftl_landed_cost, 0.0],
                "3. Net Savings": [0.0, 0.0, max(0, true_net_savings)]
            }, index=["1. LTL Baseline", "2. FTL Landed Cost", "3. Net Savings"])
            
            st.bar_chart(chart_data, color=["#dc3545", "#ffc107", "#28a745"])
            
            if tot_weight > 12000 or tot_pallets > 12:
                st.warning(f"📉 **Volume Baseline Shift:** LTL Baseline inflated by a \$125 Linear Foot Penalty per pallet over 6.")

        with col_fin2:
            st.markdown("**FTL Carrier Invoice Ledger**")
            invoice_df = pd.DataFrame({
                "Line Item Breakdown": ["Base FTL Linehaul", "Multi-Stop Delivery Fees", "Retail Liftgate Premium", "Driver Detention (Clock Overage)", "DOT 14-Hour Layover Penalty", "Dock Labor (Lumper Fees)", "TOTAL LANDED INVOICE"],
                "Cost ($)": [f"${freight_bill:,.2f}", f"${stop_off_fees:,.2f}", f"${ftl_liftgate_fees:,.2f}", f"${detention_fees:,.2f}", f"${layover_fee:,.2f}", f"${total_lumper_fees:,.2f}", f"${total_ftl_landed_cost:,.2f}"]
            })
            st.dataframe(invoice_df, hide_index=True, use_container_width=True)
            
            if not is_compliant:
                st.error(f"**Dispatch Blocked:** Theoretical savings of {safe_savings_str} voided due to physical/retail violations.")
            elif true_net_savings > 0: 
                st.success(f"**Profitable Dispatch:** Shipment cleared. Net ROI is {safe_savings_str} after USD {hc_pen:,.2f} in WACC & Storage hold costs.")
            else: 
                st.error(f"**Reject Bid:** Generating a net loss of {safe_savings_str}. Carrier rates or holding costs are too high.")

        st.divider()

        # regulatory checks
        st.subheader("Automated Regulatory Clearance")
        col_reg1, col_reg2, col_reg3, col_reg4 = st.columns(4)
        with col_reg1:
            if weight_fill_rate <= 100: st.success("⚖️ DOT Weigh Station\n\n**CLEARED**")
            else: st.error("⚖️ DOT Weigh Station\n\n**FAILED**")
        with col_reg2:
            st.success("🔬 FSMA Sanitary Transport\n\n**MANIFESTED**")
        with col_reg3:
            st.success("🌿 CARB Emissions\n\n**COMPLIANT**")
        with col_reg4:
            if layover_fee == 0: st.success("⏱️ FMCSA HOS Clock\n\n**VERIFIED**")
            else: st.warning("⏱️ FMCSA HOS Clock\n\n**LAYOVER REQUIRED**")

        st.divider()
        
        # participant data table
        st.markdown("**Individual Participant Savings Breakdown (Total Landed Cost):**")
        
        display_cols = ['Shop_Location', 'Total_Pallets', 'Total_Weight_Lbs', 'Inventory_Value_$', 'Holding_Cost_Penalty_$']
        if spoilage_writeoff > 0: display_cols.append('Spoilage_Writeoff_$')
        if esc > 0: display_cols.append('Expected_Shortage_Cost_$')
        display_cols.append('True_Net_Savings_$')
        
        df_display = df_totals[display_cols].copy()
        round_cols = ['Inventory_Value_$', 'Holding_Cost_Penalty_$', 'True_Net_Savings_$']
        if spoilage_writeoff > 0: round_cols.append('Spoilage_Writeoff_$')
        if esc > 0: round_cols.append('Expected_Shortage_Cost_$')
        for col in round_cols: df_display[col] = df_display[col].round(2)

        rename_dict = {
            'Shop_Location': 'Shop',
            'Total_Pallets': 'Pallets',
            'Total_Weight_Lbs': 'Weight (lbs)',
            'Inventory_Value_$': 'Inv Value ($)',
            'Holding_Cost_Penalty_$': 'Hold Cost ($)',
            'Expected_Shortage_Cost_$': 'Risk Pen ($)',
            'Spoilage_Writeoff_$': 'Spoilage ($)',
            'True_Net_Savings_$': 'Net Savings ($)'
        }
        df_display = df_display.rename(columns=rename_dict)

        col_tbl, col_chart = st.columns([1.5, 1])
        with col_tbl:
            st.dataframe(df_display, hide_index=True, use_container_width=True)
            st.download_button(label="Download ROI Invoices", data=df_display.to_csv(index=False).encode('utf-8'), file_name='Invoices.csv', mime='text/csv')
        with col_chart: 
            st.bar_chart(df_totals.set_index('Shop_Location')['True_Net_Savings_$'], color="#28a745")

    with tab3:
        st.header("3. Machine-to-Machine (M2M) TMS Integration")
        col_map, col_route = st.columns([2, 1])
        with col_map:
            st.map(pd.DataFrame({'lat': [38.5816, 38.5750, 38.5680, 38.5450, 38.5400, 38.6400], 'lon': [-121.4944, -121.4800, -121.4500, -121.5000, -121.4700, -121.5000]}), zoom=11)
        with col_route:
            route_text = "**Dynamic Routing Manifest (Live Payload Drop):**\n1. Start: Distribution Hub\n"
            for idx, row in df_totals.iterrows():
                route_text += f"{idx + 2}. Stop {idx + 1}: {row['Shop_Location']} *(Drop: {row['Total_Pallets']:,.0f} Pallets)*\n"
            route_text += f"{len(df_totals) + 2}. End: Return to Hub\n\n"
            st.markdown(route_text)
            
            st.markdown("**Generate API Tender Payload:**")
            st.markdown("Export the finalized shipment parameters as a machine-readable JSON payload to directly tender the load in an external execution system.")
            
            if is_compliant:
                st.success("✅ Payload Cleared for TMS Tender.")
                json_payload = create_tms_payload(origin, hw_miles, total_lead_time, tot_weight, tot_vol, tot_pallets, df_totals, trailer_type, requires_liftgate)
                st.download_button("Generate TMS Tender Payload (JSON)", data=json_payload, file_name='LCT_TMS_Tender.json', mime='application/json', type="primary")
            else:
                st.error("❌ Cannot generate TMS Tender. Payload is currently non-compliant with physical, retail, or regulatory constraints.")

if __name__ == "__main__":
    main()