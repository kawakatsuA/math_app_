import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="万能関数解析ツール", layout="centered")

# --- CSSで入力欄（number_input）およびセレクトボックス（selectbox）の幅を極小化 ---
st.markdown(
    """
    <style>
    /* 数値入力ボックスの幅と余白を詰める */
    div[data-testid="stNumberInput"] {
        width: 80px !important;
        min-width: 80px !important;
    }
    
    /* セレクトボックス（次数・種類）の幅を極小化 */
    div[data-testid="stSelectbox"] {
        width: 100px !important;
        min-width: 100px !important;
    }

    /* 列（column）の余白（ギャップ）を最小限にする */
    div[data-testid="column"] {
        padding: 0px 2px !important;
        flex: unset !important;
    }

    /* テキストの配置調整 */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[role="combobox"] {
        text-align: center;
        padding: 2px 4px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("万能関数解析 & グラフ描画ツール")

# タブの作成
tab1, tab2 = st.tabs(["📈 N次関数（多項式）", "🌊 三角関数"])

# ==========================================
# タブ 1: N次関数（1次〜5次）
# ==========================================
with tab1:
    st.header("N次関数の解析")

    # 次数選択欄も細分割コラムで幅を絞る
    sel_col, _ = st.columns([1, 11])
    with sel_col:
        degree = st.selectbox(
            "次数",
            [1, 2, 3, 4, 5],
            index=1,
            key="poly_degree",
        )

    superscripts = {1: "", 2: "²", 3: "³", 4: "⁴", 5: "⁵"}
    st.write(f"【{degree}次関数の係数（整数）を入力】")

    coeffs = []
    cols = st.columns(12)

    for i in range(degree, -1, -1):
        col_idx = degree - i
        with cols[col_idx]:
            if i > 1:
                label = f"x{superscripts[i]}"
            elif i == 1:
                label = "x"
            else:
                label = "定数"

            default_val = 1 if i == degree else (0 if i > 0 else -1)
            val = st.number_input(
                label, value=default_val, step=1, key=f"poly_coeff_{i}"
            )
            coeffs.append(val)

    st.markdown("---")

    if coeffs[0] == 0:
        st.error(
            f"⚠️ 最高次数の係数（x{superscripts[degree]}）に 0 を指定すると {degree}次関数になりません。"
        )
    else:
        # 1. 数式の組み立て
        terms = []
        for idx, (p, c) in enumerate(zip(range(degree, -1, -1), coeffs)):
            if c == 0:
                continue
            abs_c = abs(c)
            sign = (
                ("-" if c < 0 else "")
                if idx == 0
                else (" + " if c > 0 else " - ")
            )

            if p == 0:
                var = ""
                coeff_str = f"{abs_c}"
            else:
                var = f"x{superscripts[p]}" if p in superscripts else f"x^{p}"
                coeff_str = "" if abs_c == 1 else f"{abs_c}"

            terms.append(f"{sign}{coeff_str}{var}")

        eq_str = "y = " + ("".join(terms) if terms else "0")
        st.subheader("【関数の式】")
        st.latex(eq_str)

        # 2. 解の計算 (y = 0)
        st.subheader("【方程式の解 (y = 0 のとき)】")
        roots = np.roots(coeffs)
        real_roots = []
        for i, root in enumerate(roots, 1):
            if abs(root.imag) < 1e-8:
                st.write(f"**x{i} = {root.real:.4g}** (実数解)")
                real_roots.append(root.real)
            else:
                sign_str = "+" if root.imag > 0 else "-"
                st.write(
                    f"**x{i} = {root.real:.4g} {sign_str} {abs(root.imag):.4g}i** (虚数解)"
                )

        # 3. 極値（微分 f'(x) = 0）
        st.subheader("【極値（山・谷）】")
        poly = np.poly1d(coeffs)
        deriv = np.polyder(poly)
        crit_points = np.roots(deriv)

        extrema_x, extrema_y = [], []
        for pt in crit_points:
            if abs(pt.imag) < 1e-8:
                x_val = pt.real
                y_val = poly(x_val)
                extrema_x.append(x_val)
                extrema_y.append(y_val)
                st.write(f"極値の座標: **({x_val:.4g}, {y_val:.4g})**")

        if not extrema_x:
            st.write("実数の範囲に極値はありません。")

        # 4. グラフ描画
        st.subheader("【グラフ】")
        all_points = real_roots + extrema_x
        if all_points:
            min_p, max_p = min(all_points), max(all_points)
            margin = max((max_p - min_p) * 0.3, 3.0)
            x_range = (min_p - margin, max_p + margin)
        else:
            x_range = (-5, 5)

        x = np.linspace(x_range[0], x_range[1], 600)
        y = poly(x)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, y, label=eq_str, color="#2ca02c", linewidth=2)
        if extrema_x:
            ax.plot(extrema_x, extrema_y, "ro", label="極値", markersize=6)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best")
        st.pyplot(fig)


# ==========================================
# タブ 2: 複雑な三角関数（合成・足し算対応）
# ==========================================
with tab2:
    st.header("合成三角関数の解析")

    # 項の数を選択（1項〜4項）
    n_terms = st.selectbox("合成する項の数", [1, 2, 3, 4], index=1, key="n_terms")

    st.write(
        "【 $y = \\sum (A_i \\cdot f_i(B_i x + C_i)) + D$ のパラメータ設定】"
    )

    terms_data = []

    # 項の数だけ入力欄を並べて作成
    for i in range(n_terms):
        st.caption(f"■ 第 {i+1} 項")
        cols = st.columns(12)

        # 2項目以降は 符号 (＋ / −) も選択可能に
        sign = 1
        col_start = 0
        if i > 0:
            with cols[0]:
                sign_str = st.selectbox(
                    "符号", ["+", "-"], key=f"trig_sign_{i}"
                )
                sign = 1 if sign_str == "+" else -1
            col_start = 1

        with cols[col_start]:
            func_type = st.selectbox(
                "関数", ["sin", "cos", "tan"], key=f"trig_func_{i}"
            )
        with cols[col_start + 1]:
            A = st.number_input("A", value=1, step=1, key=f"trig_A_{i}")
        with cols[col_start + 2]:
            B = st.number_input("B", value=1, step=1, key=f"trig_B_{i}")
        with cols[col_start + 3]:
            C = st.number_input("C", value=0, step=1, key=f"trig_C_{i}")

        terms_data.append(
            {"sign": sign, "func": func_type, "A": A, "B": B, "C": C}
        )

    # 全体にかかる定数項 D
    st.caption("■ 定数項 D（全体シフト）")
    col_d, _ = st.columns([1, 11])
    with col_d:
        D = st.number_input("D", value=0, step=1, key="trig_D_total")

    st.markdown("---")

    # B に 0 が入っていないか判定
    if any(term["B"] == 0 for term in terms_data):
        st.error("⚠️ B (周期倍率) に 0 を指定することはできません。")
    else:
        # 1. LaTeX 数式の組み立て
        latex_terms = []
        for i, term in enumerate(terms_data):
            s = ""
            if i > 0:
                s = " + " if term["sign"] == 1 else " - "
            elif term["sign"] == -1:
                s = "-"

            coeff_val = abs(term["A"])
            coeff_str = "" if coeff_val == 1 else f"{coeff_val}"

            f_name = term["func"]
            b_val = term["B"]
            c_val = term["C"]

            # (Bx + C) の組み立て
            inside = f"{b_val}x" if b_val != 1 else "x"
            if c_val > 0:
                inside += f" + {c_val}"
            elif c_val < 0:
                inside += f" - {abs(c_val)}"

            latex_terms.append(f"{s}{coeff_str}\\{f_name}({inside})")

        if D > 0:
            latex_terms.append(f" + {D}")
        elif D < 0:
            latex_terms.append(f" - {abs(D)}")

        eq_str = "y = " + "".join(latex_terms)

        st.subheader("【合成関数の式】")
        st.latex(eq_str)

        # 2. グラフの計算 & 描画
        st.subheader("【グラフ】")

        # 表示範囲（-4π 〜 +4π）
        x = np.linspace(-4 * np.pi, 4 * np.pi, 2000)
        y = np.full_like(x, float(D))

        has_tan = False
        for term in terms_data:
            A_eff = term["sign"] * term["A"]
            B_val = term["B"]
            C_val = term["C"]

            if term["func"] == "sin":
                y += A_eff * np.sin(B_val * x + C_val)
            elif term["func"] == "cos":
                y += A_eff * np.cos(B_val * x + C_val)
            else:  # tan
                has_tan = True
                tan_y = A_eff * np.tan(B_val * x + C_val)
                y += tan_y

        # tanの漸近線（無限大飛び）を破棄
        if has_tan:
            y[np.abs(y) > 20] = np.nan

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, y, label="y", color="#ff7f0e", linewidth=2)

        ax.axhline(D, color="gray", linewidth=0.8, linestyle=":")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

        if has_tan:
            ax.set_ylim(-10 + D, 10 + D)

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlabel("x (rad)")
        ax.set_ylabel("y")
        st.pyplot(fig)