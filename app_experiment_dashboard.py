import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Пилот «Пульс модуля»: Эксперимент и Продуктовый радар",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("pulse_module_experiment_synthetic_data.csv")

df_raw = load_data()

# ==========================================
# САЙДБАР: ФИЛЬТРЫ И УПРАВЛЕНИЕ ВЫБОРКОЙ
# ==========================================
st.sidebar.title("🎛 Панель управления пилотом")

data_mode = st.sidebar.radio(
    "Режим фильтрации шума:",
    options=["Только чистые анкеты (S_noise < 2)", "Все анкеты (Raw Data)"],
    index=0,
    help="Отсекает анкеты с S_noise >= 2 (Straightlining, First-Option Bias, Logic Conflict)"
)

courses_list = ["Все пилотные курсы (Пул 5 потоков)"] + sorted(df_raw["course_name"].unique().tolist())
selected_course = st.sidebar.selectbox("Курс / Программа:", courses_list)

if selected_course == "Все пилотные курсы (Пул 5 потоков)":
    df_course = df_raw.copy()
else:
    df_course = df_raw[df_raw["course_name"] == selected_course].copy()

cohorts_avail = sorted(df_course["cohort_id"].unique().tolist())
selected_cohorts = st.sidebar.multiselect("Пилотные потоки:", cohorts_avail, default=cohorts_avail)

modules_avail = sorted(df_course["module_id"].unique().tolist())
selected_modules = st.sidebar.multiselect("Модули:", modules_avail, default=modules_avail)

if not selected_cohorts or not selected_modules:
    st.warning("Выберите хотя бы один поток и модуль в панели фильтров.")
    st.stop()

# Срез до очистки (для расчета GRR и воронки охвата)
df_slice_raw = df_course[
    (df_course["cohort_id"].isin(selected_cohorts)) &
    (df_course["module_id"].isin(selected_modules))
].copy()

# Срез с учетом переключателя очистки (для продуктовых метрик)
if data_mode == "Только чистые анкеты (S_noise < 2)":
    df_active = df_slice_raw[df_slice_raw["is_garbage"] == 0].copy()
else:
    df_active = df_slice_raw.copy()

# ==========================================
# РАСЧЕТ МЕТРИК ЭКСПЕРИМЕНТА (DECISION GATES)
# ==========================================
# Генеральная совокупность N по выбранным потокам и модулям (сумма N когорт * число выбранных модулей или уникальные студенты)
cohort_N_map = df_slice_raw.groupby("cohort_id")["cohort_size_N"].first().to_dict()
total_enrolled_N = sum(cohort_N_map.values())
# Потенциальная емкость анкет с учетом числа выбранных модулей
total_expected_surveys = total_enrolled_N * len(selected_modules)

n_raw = len(df_slice_raw)
n_garbage = int(df_slice_raw["is_garbage"].sum())
n_clean = n_raw - n_garbage

grr_pct = (n_garbage / n_raw * 100) if n_raw > 0 else 0.0
raw_coverage_pct = (n_raw / total_expected_surveys * 100) if total_expected_surveys > 0 else 0.0
clean_coverage_pct = (n_clean / total_expected_surveys * 100) if total_expected_surveys > 0 else 0.0

# Расчет MoE (Margin of Error) с поправкой на конечную популяцию (FPC)
Z = 1.96
p = 0.5
if n_clean > 0 and total_expected_surveys > 1:
    fpc = np.sqrt(max(0, (total_expected_surveys - n_clean) / (total_expected_surveys - 1)))
    moe_pct = Z * np.sqrt((p * (1 - p)) / n_clean) * fpc * 100
else:
    moe_pct = 100.0

# Расчет RR_overdue (Относительный риск долгов на следующем модуле)
df_rr = df_slice_raw[(df_slice_raw["is_garbage"] == 0) & (df_slice_raw["next_module_overdue"].notna())]
if len(df_rr) > 0:
    alert_group = df_rr[df_rr["has_mhi_alert"] == 1]
    base_group = df_rr[df_rr["has_mhi_alert"] == 0]
    risk_alert = alert_group["next_module_overdue"].mean() if len(alert_group) > 0 else 0.0
    risk_base = base_group["next_module_overdue"].mean() if len(base_group) > 0 else 0.0
    rr_overdue = (risk_alert / risk_base) if risk_base > 0 else 0.0
else:
    risk_alert, risk_base, rr_overdue = 0.0, 0.0, 0.0

# ==========================================
# ШАПКА ДАШБОРДА: ВОРОНКА ОХВАТА И ОБЪЕМА
# ==========================================
st.title("📊 Дашборд эксперимента: Пилотный запуск «Пульс модуля»")
st.caption(f"Периметр: **{selected_course}** | Выбрано потоков: **{len(selected_cohorts)}** | Режим: **{data_mode}**")

top1, top2, top3, top4, top5 = st.columns(5)
top1.metric("Студентов на потоках (N)", f"{total_enrolled_N} чел.", f"Емкость среза: {total_expected_surveys} анкет")
top2.metric("Собрано анкет (n raw)", f"{n_raw} шт.", f"Охват: {raw_coverage_pct:.1f}%")
top3.metric("Чистых анкет (n clean)", f"{n_clean} шт.", f"Чистый охват: {clean_coverage_pct:.1f}%")
top4.metric("Отбраковано шума (GRR)", f"{grr_pct:.1f}%", f"{n_garbage} мусорных анкет", delta_color="inverse" if grr_pct > 15 else "normal")
top5.metric("Погрешность выборки (MoE)", f"±{moe_pct:.1f}%", "Цель ≤ ±7.0%", delta_color="inverse" if moe_pct > 10 else "normal")

st.divider()

# ==========================================
# НАВИГАЦИЯ ПО ВКЛАДКАМ
# ==========================================
tab_exp, tab_mhi, tab_csi, tab_ops = st.tabs([
    "🧪 1. Валидация эксперимента (Gates 1–3)",
    "🟢 2. Пульс здоровья модуля (MHI)",
    "🟡 3. Унаследованные метрики (CSI 1–5)",
    "🚨 4. Closed-Loop & Техмониторинг (№2920)"
])

# ------------------------------------------
# ВКЛАДКА 1: ВАЛИДАЦИЯ ЭКСПЕРИМЕНТА
# ------------------------------------------
with tab_exp:
    st.subheader("Матрица контрольных ворот пилота (Decision Gates)")
    
    gate1_status = "🟢 Успех (<15%)" if grr_pct < 15 else ("🟡 Доработка (15–30%)" if grr_pct <= 30 else "🔴 Провал (>30%)")
    gate2_status = "🟢 Успех (Охват ≥65%)" if clean_coverage_pct >= 65 else ("🟡 Доработка (45–64%)" if clean_coverage_pct >= 45 else "🔴 Провал (<45%)")
    gate3_status = "🟢 Успех (RR ≥ 2.5)" if rr_overdue >= 2.5 else ("🟡 Доработка (2.0–2.49)" if rr_overdue >= 2.0 else "🔴 Провал (<1.5)")
    
    scorecard_df = pd.DataFrame([
        {
            "Контрольные ворота": "Gate 1. Безопасность (GRR)",
            "Что проверяем": "Доля мусорных прокликов (S_noise ≥ 2)",
            "Текущий факт": f"{grr_pct:.1f}% ({n_garbage} из {n_raw})",
            "Критерий успеха": "GRR < 15%",
            "Статус": gate1_status
        },
        {
            "Контрольные ворота": "Gate 2. Достоверность (MoE & Охват)",
            "Что проверяем": "Чистый охват когорты и предельная погрешность",
            "Текущий факт": f"Охват {clean_coverage_pct:.1f}% | MoE ±{moe_pct:.1f}% (n={n_clean})",
            "Критерий успеха": "Чистый охват ≥ 65% (MoE ≤ 7% в пуле)",
            "Статус": gate2_status
        },
        {
            "Контрольные ворота": "Gate 3. Бизнес-ценность (RR overdue)",
            "Что проверяем": "Во сколько раз алерт MHI повышает риск долгов в Модуле N+1",
            "Текущий факт": f"RR = {rr_overdue:.2f}x ({risk_alert*100:.1f}% vs {risk_base*100:.1f}%)",
            "Критерий успеха": "RR ≥ 2.5x",
            "Статус": gate3_status
        }
    ])
    st.dataframe(scorecard_df, use_container_width=True, hide_index=True)
    
    col_g1, col_g2, col_g3 = st.columns(3)
    
    with col_g1:
        st.markdown("##### 🔍 Gate 1: Детекция шума ($S_{noise}$)")
        noise_breakdown = pd.DataFrame({
            "Индикатор": [
                "Straightlining (SD CSI = 0)",
                "First-Option Bias (rushed+clear+full)",
                "Logic Conflict (Алерт MHI + все 5 в CSI)",
                "Итого мусор (S_noise ≥ 2)"
            ],
            "Анкет": [
                int(df_slice_raw["I_straight"].sum()),
                int(df_slice_raw["I_first_option"].sum()),
                int(df_slice_raw["I_conflict"].sum()),
                n_garbage
            ]
        })
        noise_breakdown["% от всех анкет"] = (noise_breakdown["Анкет"] / n_raw * 100).round(1)
        fig_noise = px.bar(
            noise_breakdown, x="% от всех анкет", y="Индикатор", orientation="h",
            text="Анкет", color="Индикатор",
            color_discrete_sequence=["#FFA726", "#FB8C00", "#EF5350", "#C62828"]
        )
        fig_noise.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_noise, use_container_width=True)
        
    with col_g2:
        st.markdown("##### 📐 Gate 2: Сжатие погрешности (MoE)")
        # Кривая сжатия MoE по мере накопления анкет
        n_range = np.arange(10, max(250, n_clean + 50), 5)
        moe_curve = [1.96 * np.sqrt(0.25 / x) * 100 for x in n_range]
        fig_moe = go.Figure()
        fig_moe.add_trace(go.Scatter(x=n_range, y=moe_curve, mode="lines", name="Теоретическая MoE", line=dict(color="#1E88E5")))
        fig_moe.add_hline(y=7.0, line_dash="dash", line_color="#43A047", annotation_text="Целевой порог ±7%")
        fig_moe.add_trace(go.Scatter(
            x=[n_clean], y=[moe_pct], mode="markers+text",
            name="Текущий срез", text=[f"n={n_clean} (±{moe_pct:.1f}%)"],
            textposition="top right", marker=dict(color="#E53935", size=12)
        ))
        fig_moe.update_layout(height=280, xaxis_title="Объем чистых анкет (n)", yaxis_title="MoE (±%)", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_moe, use_container_width=True)
        
    with col_g3:
        st.markdown("##### 🎯 Gate 3: Прогноз долгов ($RR_{overdue}$)")
        rr_df = pd.DataFrame({
            "Группа в Модуле N": ["С красным алертом MHI", "Без алертов (Норма)"],
            "Доля должников в Модуле N+1 (%)": [risk_alert * 100, risk_base * 100]
        })
        fig_rr = px.bar(
            rr_df, x="Группа в Модуле N", y="Доля должников в Модуле N+1 (%)",
            color="Группа в Модуле N", text=rr_df["Доля должников в Модуле N+1 (%)"].apply(lambda x: f"{x:.1f}%"),
            color_discrete_map={"С красным алертом MHI": "#D32F2F", "Без алертов (Норма)": "#388E3C"}
        )
        fig_rr.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_rr, use_container_width=True)

# ------------------------------------------
# ВКЛАДКА 2: ПУЛЬС ЗДОРОВЬЯ МОДУЛЯ (MHI)
# ------------------------------------------
with tab_mhi:
    st.subheader("Индекс здоровья модуля (Module Health Index)")
    
    n_act = len(df_active)
    mhi_norm_cnt = int((df_active["has_mhi_alert"] == 0).sum())
    mhi_rate = (mhi_norm_cnt / n_act * 100) if n_act > 0 else 0.0
    
    rushed_pct = (df_active["pacing_score"] == "rushed").mean() * 100 if n_act > 0 else 0.0
    frag_pct = (df_active["cohesion_score"] == "fragmented").mean() * 100 if n_act > 0 else 0.0
    dep_pct = (df_active["energy_score"] == "depleted").mean() * 100 if n_act > 0 else 0.0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MHI Rate (Без алертов)", f"{mhi_rate:.1f}%", "Норма ≥ 75%", delta_color="normal" if mhi_rate >= 75 else "inverse")
    m2.metric("Спешка (Pacing: Rushed)", f"{rushed_pct:.1f}%", "Порог < 25%", delta_color="inverse" if rushed_pct > 25 else "normal")
    m3.metric("Разрыв логики (Fragmented)", f"{frag_pct:.1f}%", "Порог < 15%", delta_color="inverse" if frag_pct > 15 else "normal")
    m4.metric("Истощение (Depleted)", f"{dep_pct:.1f}%", "Порог < 20%", delta_color="inverse" if dep_pct > 20 else "normal")
    
    st.divider()
    c_p, c_c, c_e = st.columns(3)
    
    def plot_mhi_stack(df_in, col, order, colors, title):
        ct = (pd.crosstab(df_in["module_id"], df_in[col], normalize="index") * 100).reset_index()
        for cat in order:
            if cat not in ct.columns:
                ct[cat] = 0.0
        melted = ct.melt(id_vars=["module_id"], value_vars=order, var_name="Ответ", value_name="Доля (%)")
        fig = px.bar(
            melted, x="module_id", y="Доля (%)", color="Ответ",
            color_discrete_map=colors, category_orders={"Ответ": order},
            title=title, text=melted["Доля (%)"].apply(lambda v: f"{v:.0f}%" if v >= 6 else "")
        )
        fig.update_layout(barmode="stack", height=340, xaxis_title="", yaxis_title="%")
        return fig
        
    with c_p:
        st.plotly_chart(plot_mhi_stack(
            df_active, "pacing_score", ["slow", "optimal", "rushed"],
            {"slow": "#FBC02D", "optimal": "#388E3C", "rushed": "#D32F2F"},
            "1. Ритм и темп (Pacing)"
        ), use_container_width=True)
    with c_c:
        st.plotly_chart(plot_mhi_stack(
            df_active, "cohesion_score", ["fragmented", "confused", "clear"],
            {"fragmented": "#D32F2F", "confused": "#FBC02D", "clear": "#388E3C"},
            "2. Связность материалов (Cohesion)"
        ), use_container_width=True)
    with c_e:
        st.plotly_chart(plot_mhi_stack(
            df_active, "energy_score", ["depleted", "tired", "full"],
            {"depleted": "#D32F2F", "tired": "#FBC02D", "full": "#388E3C"},
            "3. Запас сил (Energy)"
        ), use_container_width=True)

# ------------------------------------------
# ВКЛАДКА 3: УНАСЛЕДОВАННЫЕ МЕТРИКИ (CSI)
# ------------------------------------------
with tab_csi:
    st.subheader("Унаследованные вопросы CSI (6 метрик, шкала 1–5)")
    csi_map = {
        "csi_materials": "4. Качество учебных материалов (лонгриды, видео, скринкасты)",
        "csi_tasks": "5. Качество заданий (тесты, квизы, самостоятельные задания)",
        "csi_experts": "7. Работа экспертов / преподавателей в модуле",
        "csi_support_speed": "8. Скорость работы команды сопровождения (SLA)",
        "csi_support_care": "9. Забота и поддержка команды сопровождения",
        "csi_platform": "10. Удобство сайта / платформы в рамках модуля"
    }
    
    csi_rows = []
    for col, label in csi_map.items():
        s = df_active[col]
        top2 = (s >= 4).mean() * 100
        bot2 = (s <= 2).mean() * 100
        csi_rows.append({
            "Вопрос CSI": label,
            "Top-2 Box (4–5)": f"{top2:.1f}%",
            "Bottom-2 Box (1–2)": f"{bot2:.1f}%",
            "Среднее": f"{s.mean():.2f}",
            "Медиана": f"{s.median():.1f}"
        })
    st.dataframe(pd.DataFrame(csi_rows), use_container_width=True, hide_index=True)

# ------------------------------------------
# ВКЛАДКА 4: CLOSED-LOOP & ТЕХМОНИТОРИНГ
# ------------------------------------------
with tab_ops:
    col_ops1, col_ops2 = st.columns([3, 2])
    
    with col_ops1:
        st.subheader("🚨 Реестр студентов для интервенции тьюторов")
        alerts_df = df_active[df_active["has_mhi_alert"] == 1][[
            "submitted_at", "cohort_id", "module_id", "student_id",
            "pacing_score", "cohesion_score", "energy_score", "next_module_overdue", "open_feedback_2920"
        ]].copy()
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)
        
    with col_ops2:
        st.subheader("🛠 Мониторинг бага обязательности (Вопрос №2920)")
        dummy_cnt = int(df_slice_raw["is_dummy_feedback_2920"].sum())
        dummy_pct = (dummy_cnt / n_raw * 100) if n_raw > 0 else 0.0
        st.metric(
            "Доля фиктивных символов в открытом поле (., -, ?, +)",
            f"{dummy_pct:.1f}% ({dummy_cnt} анкет)",
            delta="Баг обязательности поля в LMS",
            delta_color="inverse"
        )
        st.caption("В Срезе №1 выявлено, что вопрос №2920 в LMS ошибочно настроен как обязательный, из-за чего ~27.8% студентов вводят одиночные знаки препинания для разблокировки экрана.")
