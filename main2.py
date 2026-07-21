import flet as ft
import matplotlib
matplotlib.use('Agg')         
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

# ---------- توابع کمکی ----------
def safe_eval(expr, x):
    allowed = {
        "x": x, "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": abs,
        "pi": np.pi, "e": np.e
    }
    try:
        return eval(expr, {"__builtins__": {}}, allowed)
    except Exception:
        return None

def numerical_derivative(f_expr, x, h=1e-6):
    fx = safe_eval(f_expr, x)
    fxh = safe_eval(f_expr, x + h)
    if fx is None or fxh is None:
        return None
    return (fxh - fx) / h

# ---------- بخش ریشه‌یابی ----------
def root_bisection(func_str, a, b, tol, max_iter):
    points = []
    fa = safe_eval(func_str, a)
    fb = safe_eval(func_str, b)
    if fa is None or fb is None or fa*fb > 0:
        return None, points, "f(a) و f(b) هم‌علامت هستند یا تابع تعریف‌نشده."
    for i in range(max_iter):
        c = (a + b) / 2
        fc = safe_eval(func_str, c)
        if fc is None:
            return None, points, "تابع در نقطه‌ای تعریف‌نشده."
        err = abs(b - a)
        points.append((c, fc))
        if err < tol:
            return c, points, None
        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc
    return None, points, "به حداکثر تکرار رسید."

def root_false_position(func_str, a, b, tol, max_iter):
    points = []
    fa = safe_eval(func_str, a)
    fb = safe_eval(func_str, b)
    if fa is None or fb is None or fa*fb > 0:
        return None, points, "f(a) و f(b) هم‌علامت یا تعریف‌نشده."
    for i in range(max_iter):
        c = b - fb * (a - b) / (fa - fb) if fa != fb else (a+b)/2
        fc = safe_eval(func_str, c)
        if fc is None:
            return None, points, "تابع تعریف‌نشده."
        err = abs(fc)
        points.append((c, fc))
        if abs(fc) < tol:
            return c, points, None
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return None, points, "به حداکثر تکرار رسید."

def root_newton(func_str, x0, tol, max_iter):
    points = []
    x = x0
    for i in range(max_iter):
        fx = safe_eval(func_str, x)
        fpx = numerical_derivative(func_str, x)
        if fx is None or fpx is None or fpx == 0:
            return None, points, "مشتق صفر یا تابع تعریف‌نشده."
        x_new = x - fx / fpx
        err = abs(x_new - x)
        points.append((x_new, safe_eval(func_str, x_new)))
        if err < tol:
            return x_new, points, None
        x = x_new
    return None, points, "به حداکثر تکرار رسید."

def root_secant(func_str, x0, x1, tol, max_iter):
    points = []
    for i in range(max_iter):
        f0 = safe_eval(func_str, x0)
        f1 = safe_eval(func_str, x1)
        if f0 is None or f1 is None or f1-f0 == 0:
            return None, points, "تابع تعریف‌نشده یا مخرج صفر."
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        err = abs(x2 - x1)
        points.append((x2, safe_eval(func_str, x2)))
        if err < tol:
            return x2, points, None
        x0, x1 = x1, x2
    return None, points, "به حداکثر تکرار رسید."

# ---------- بخش انتگرال‌گیری ----------
def composite_trapezoidal(func_str, a, b, n):
    h = (b - a) / n
    xs = [a + i*h for i in range(n+1)]
    ys = [safe_eval(func_str, x) for x in xs]
    if any(y is None for y in ys):
        return None, None, None
    integral = h * (0.5*ys[0] + sum(ys[1:-1]) + 0.5*ys[-1])
    return integral, xs, ys

def composite_simpson13(func_str, a, b, n):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    xs = [a + i*h for i in range(n+1)]
    ys = [safe_eval(func_str, x) for x in xs]
    if any(y is None for y in ys):
        return None, None, None
    sum_odd = sum(ys[i] for i in range(1, n, 2))
    sum_even = sum(ys[i] for i in range(2, n-1, 2))
    integral = h/3 * (ys[0] + 4*sum_odd + 2*sum_even + ys[n])
    return integral, xs, ys

def composite_simpson38(func_str, a, b, n):
    if n % 3 != 0:
        n = ((n//3)+1)*3
    h = (b - a) / n
    xs = [a + i*h for i in range(n+1)]
    ys = [safe_eval(func_str, x) for x in xs]
    if any(y is None for y in ys):
        return None, None, None
    integral = 3*h/8 * (ys[0] + ys[n] +
                        3*sum(ys[i] for i in range(1, n) if i%3 != 0) +
                        2*sum(ys[i] for i in range(3, n-2, 3)))
    return integral, xs, ys

# ---------- رسم نمودار ----------
def plot_root(func_str, points, a_plot, b_plot, method_name):
    x_vals = np.linspace(a_plot, b_plot, 400)
    y_vals = [safe_eval(func_str, x) for x in x_vals]
    valid = [(x,y) for x,y in zip(x_vals, y_vals) if y is not None]
    if not valid:
        return None
    xs, ys = zip(*valid)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(xs, ys, label=f"f(x) = {func_str}")
    ax.axhline(0, color='gray', linewidth=0.8)
    iters_x = [p[0] for p in points if p[0] is not None]
    iters_y = [safe_eval(func_str, x) for x in iters_x]
    ax.scatter(iters_x, iters_y, color='red', zorder=5)
    for i, (ix, iy) in enumerate(zip(iters_x, iters_y)):
        ax.annotate(str(i), (ix, iy), textcoords="offset points", xytext=(5,5), fontsize=8, color='red')
    ax.set_title(f"روش: {method_name}")
    ax.legend(); ax.grid(True); fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img

def plot_integration(func_str, a, b, xs, ys, method_name):
    fig, ax = plt.subplots(figsize=(7,4))
    x_curve = np.linspace(a, b, 300)
    y_curve = [safe_eval(func_str, x) for x in x_curve]
    ax.plot(x_curve, y_curve, label=f"f(x) = {func_str}")
    ax.fill_between(xs, ys, alpha=0.3, color='orange')
    ax.scatter(xs, ys, s=15, color='red')
    ax.set_title(f"انتگرال‌گیری به روش {method_name}")
    ax.legend(); ax.grid(True); fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img

# ---------- رابط کاربری ----------
def main(page: ft.Page):
    page.title = "محاسبات عددی - ریشه‌یابی و انتگرال‌گیری"
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # هدر با نام استاد
    header = ft.Container(
        content=ft.Column([
            ft.Text("پروژه محاسبات عددی", size=24, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
            ft.Text("استاد راهنما: سمانه گلستانی", size=18, italic=True, color=ft.colors.BLUE_700, text_align=ft.TextAlign.CENTER)
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        margin=ft.margin.only(bottom=20)
    )

    # --- انتخاب حالت ---
    mode_dd = ft.Dropdown(
        label="حالت",
        options=[
            ft.dropdown.Option("root", "ریشه‌یابی معادلات"),
            ft.dropdown.Option("integ", "انتگرال‌گیری عددی"),
        ],
        value="root",
        width=250
    )

    # --- ورودی‌های مشترک ---
    func_input = ft.TextField(label="تابع f(x)", value="x**3 - 2*x - 5", width=300)

    # --- ورودی‌های ریشه‌یابی ---
    method_root_dd = ft.Dropdown(
        label="روش",
        options=[
            ft.dropdown.Option("bisection", "دو بخشی"),
            ft.dropdown.Option("false_position", "نابجایی"),
            ft.dropdown.Option("newton", "نیوتون-رافسون"),
            ft.dropdown.Option("secant", "وتر"),
        ],
        value="bisection", width=200
    )
    a_root = ft.TextField(label="a (کران چپ/حدس)", value="2", width=140)
    b_root = ft.TextField(label="b (کران راست/حدس دوم)", value="3", width=140)
    tol_root = ft.TextField(label="خطای مجاز", value="1e-6", width=140)
    max_iter_root = ft.TextField(label="حداکثر تکرار", value="50", width=140)
    root_panel = ft.Column([
        method_root_dd,
        ft.Row([a_root, b_root, tol_root, max_iter_root], alignment=ft.MainAxisAlignment.CENTER)
    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    # --- ورودی‌های انتگرال ---
    method_int_dd = ft.Dropdown(
        label="روش",
        options=[
            ft.dropdown.Option("trap", "ذوزنقه‌ای"),
            ft.dropdown.Option("simp13", "سیمپسون ۱/۳"),
            ft.dropdown.Option("simp38", "سیمپسون ۳/۸"),
        ],
        value="trap", width=200
    )
    a_int = ft.TextField(label="a", value="0", width=140)
    b_int = ft.TextField(label="b", value="1", width=140)
    n_int = ft.TextField(label="تعداد زیربازه (n)", value="10", width=140)
    int_panel = ft.Column([
        method_int_dd,
        ft.Row([a_int, b_int, n_int], alignment=ft.MainAxisAlignment.CENTER)
    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    # --- خروجی‌ها ---
    plot_img = ft.Image(src="", width=700, height=400, fit=ft.ImageFit.CONTAIN)
    plot_container = ft.Container(content=plot_img, alignment=ft.alignment.center, margin=ft.margin.only(top=10, bottom=10))

    result_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("مرحله")),
            ft.DataColumn(ft.Text("x")),
            ft.DataColumn(ft.Text("f(x)")),
            ft.DataColumn(ft.Text("خطا")),
        ],
        rows=[]
    )
    result_text = ft.Text("", size=16, weight=ft.FontWeight.BOLD)
    status_text = ft.Text("", color=ft.colors.RED)

    # --- کنترل پنل‌ها (در یک container وسط) ---
    input_container = ft.Column([func_input, root_panel], horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    def mode_changed(e):
        if mode_dd.value == "root":
            input_container.controls = [func_input, root_panel]
            result_table.visible = True
            result_text.value = ""
            status_text.value = ""
            plot_img.src_base64 = ""
        else:
            input_container.controls = [func_input, int_panel]
            result_table.visible = False
            result_text.value = ""
            status_text.value = ""
            plot_img.src_base64 = ""
        page.update()

    mode_dd.on_change = mode_changed

    # --- دکمه‌ها ---
    def solve_root(e):
        func_str = func_input.value.strip()
        method = method_root_dd.value
        try:
            a0 = float(a_root.value)
            b0 = float(b_root.value)
            tol = float(tol_root.value)
            max_iter = int(max_iter_root.value)
        except:
            status_text.value = "مقادیر عددی نامعتبر."
            page.update()
            return
        result_table.rows.clear()
        status_text.value = ""
        result_text.value = ""
        points = []
        root = None
        msg = ""

        if method == "bisection":
            root, points, msg = root_bisection(func_str, a0, b0, tol, max_iter)
        elif method == "false_position":
            root, points, msg = root_false_position(func_str, a0, b0, tol, max_iter)
        elif method == "newton":
            root, points, msg = root_newton(func_str, a0, tol, max_iter)
        elif method == "secant":
            root, points, msg = root_secant(func_str, a0, b0, tol, max_iter)

        for i, (x, fx) in enumerate(points):
            err = abs(b0 - a0) if i==0 and method in ("bisection","false_position") else (abs(points[i][0]-points[i-1][0]) if i>0 else 0)
            if method in ("newton","secant") and i>0:
                err = abs(points[i][0] - points[i-1][0])
            result_table.rows.append(
                ft.DataRow(cells=[
                    ft.DataCell(ft.Text(str(i+1))),
                    ft.DataCell(ft.Text(f"{x:.8f}")),
                    ft.DataCell(ft.Text(f"{safe_eval(func_str, x):.2e}")),
                    ft.DataCell(ft.Text(f"{err:.2e}")),
                ])
            )

        if root is not None:
            result_text.value = f"✅ ریشه: {root:.10f}"
            result_text.color = ft.colors.GREEN
            status_text.value = ""
        else:
            status_text.value = msg or "خطا در محاسبه."

        if method in ("bisection","false_position"):
            x_min, x_max = a0, b0
        else:
            all_x = [p[0] for p in points if p[0] is not None]
            if all_x:
                x_min, x_max = min(all_x)-1, max(all_x)+1
            else:
                x_min, x_max = a0-1, b0+1
        img = plot_root(func_str, points, x_min, x_max, method_root_dd.value)
        if img:
            plot_img.src_base64 = img
        page.update()

    def solve_integ(e):
        func_str = func_input.value.strip()
        method = method_int_dd.value
        try:
            a = float(a_int.value)
            b = float(b_int.value)
            n = int(n_int.value)
        except:
            status_text.value = "مقادیر عددی نامعتبر."
            page.update()
            return
        status_text.value = ""
        result_text.value = ""
        plot_img.src_base64 = ""

        if method == "trap":
            integral, xs, ys = composite_trapezoidal(func_str, a, b, n)
            method_name = "ذوزنقه‌ای"
        elif method == "simp13":
            integral, xs, ys = composite_simpson13(func_str, a, b, n)
            method_name = "سیمپسون ۱/۳"
        elif method == "simp38":
            integral, xs, ys = composite_simpson38(func_str, a, b, n)
            method_name = "سیمپسون ۳/۸"
        else:
            integral, xs, ys = None, None, None

        if integral is None:
            status_text.value = "خطا در محاسبه (تابع در برخی نقاط تعریف‌نشده)."
            page.update()
            return

        result_text.value = f"🔹 مقدار تقریبی انتگرال: {integral:.8f}"
        result_text.color = ft.colors.BLUE
        img = plot_integration(func_str, a, b, xs, ys, method_name)
        if img:
            plot_img.src_base64 = img
        page.update()

    solve_root_btn = ft.ElevatedButton("حل ریشه", on_click=solve_root)
    solve_int_btn = ft.ElevatedButton("محاسبه انتگرال", on_click=solve_integ)

    # چیدمان اصلی در یک ستون وسط‌چین
    main_column = ft.Column(
        [
            header,
            mode_dd,
            input_container,
            ft.Row([solve_root_btn, solve_int_btn], alignment=ft.MainAxisAlignment.CENTER),
            result_text,
            status_text,
            plot_container,
            result_table
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        alignment=ft.MainAxisAlignment.START,
        spacing=15
    )

    page.add(main_column)

    mode_changed(None)

ft.app(target=main)
