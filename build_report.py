from pathlib import Path
from html import unescape
import json

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, PageBreak, Table, TableStyle)

ROOT = Path(__file__).resolve().parent


def report_fonts():
    candidates = (
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
        (Path("/Library/Fonts/Arial.ttf"), Path("/Library/Fonts/Arial Bold.ttf")),
    )
    for normal, bold in candidates:
        if normal.is_file() and bold.is_file():
            return normal, bold
    raise FileNotFoundError("Arial or DejaVu Sans fonts are required to build the PDF")


def main():
    forecast = json.loads((ROOT / "validation/forecast.json").read_text(encoding="utf-8"))
    model = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / model["evaluation_root"] / "summary.json").read_text(encoding="utf-8"))
    runs = audit["modes"][model["evaluation_mode"]]["validation_per_repeat"]
    training = json.loads((ROOT / "validation/train_metrics.json").read_text(encoding="utf-8"))["totals"]
    normal_font, bold_font = report_fonts()
    pdfmetrics.registerFont(TTFont("Arial", str(normal_font)))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(bold_font)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")
    ink, muted, blue, pale = (colors.HexColor(s) for s in ("#172B3A", "#526171", "#145B83", "#EDF4F7"))
    styles = {
        "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=18, leading=22, textColor=ink, spaceAfter=8),
        "sub": ParagraphStyle("sub", fontName="Arial", fontSize=8.8, leading=11.5, textColor=muted, spaceAfter=12),
        "h": ParagraphStyle("h", fontName="Arial-Bold", fontSize=11, leading=14.5, textColor=blue, spaceBefore=9, spaceAfter=5),
        "p": ParagraphStyle("p", fontName="Arial", fontSize=9.5, leading=13, textColor=ink, spaceAfter=6),
        "small": ParagraphStyle("small", fontName="Arial", fontSize=8, leading=10.5, textColor=muted, spaceAfter=4),
        "metric": ParagraphStyle("metric", fontName="Arial-Bold", fontSize=12, leading=16, textColor=blue),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10.5, textColor=ink, spaceAfter=5),
    }
    story, markdown = [], []
    def p(text, style="p"):
        story.append(Paragraph(text, styles[style]))
        text = unescape(text.replace("<b>", "**").replace("</b>", "**").replace("<br/>", "\n"))
        if style == "title":
            text = ("# " if not markdown else "## ") + text
        elif style == "code":
            text = "```sh\n" + text + "\n```"
        markdown.append(text + "\n")
    def h(text):
        story.append(Paragraph(text, styles["h"]))
        markdown.append("## " + text + "\n")
    def callout(text):
        box = Table([[Paragraph(text, styles["metric"])]], colWidths=[507])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), pale),
                                ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story.append(box)
        markdown.append(text.replace("<br/>", "\n") + "\n")


    p("Постобработка OCR паспорта РФ", "title")
    p("Решение и валидация · CPU, стандартная библиотека Python, без внешних зависимостей · 25.10.2026", "sub")
    callout(f"Test: 2000 документов · {forecast['test_accepted']} принято ({100*forecast['test_accepted']/forecast['test_documents']:.1f}%) · {forecast['test_rejected']} отклонено")

    h("1. Задача и архитектура решения")
    p("Цель - максимизировать число принятых документов при отсутствии ошибочных принятий. Документ считается корректным, если все десять полей совпадают с разметкой и паспорт действителен на 01.10.2026. Алгоритм объединяет геометрические проверки, сверку зон распознавания, справочник подразделений и проверку дат.")
    p("1) <b>Геометрический контроль строк:</b> строки визуальной зоны собираются по координатам с проверкой верхней и нижней границы поля, а также межстрочных разрывов. Пропуск любой строки ведёт к отказу.")
    p("2) <b>Сверка визуальной зоны и MRZ:</b> ФИО сопоставляются по правилам кодирования кириллицы внутренних паспортов. При оценке уверенности MRZ символы '&lt;' не учитываются. В нижней строке проверяются допустимый алфавит, структура и четыре контрольные цифры. Восстанавливается четвёртая цифра серии. Ограниченные исправления разрешены при достаточном подтверждении; уверенные противоречия ведут к отказу.")
    p("3) <b>Свободные тексты:</b> орган выдачи сопоставляется со справочником DaData и названиями из обучения. Неоднозначные подстановки, изменение пунктуации и грамматических окончаний запрещены. Место рождения не достраивается: проверяются полнота рамки и уверенность символов. Запрета незнакомых имён нет.")
    p("4) <b>Проверка дат:</b> календарная корректность, выдача не ранее 14-летия и не позже даты оценки, сроки замены в 20/45 лет и 90 календарных дней после возрастной границы. Доступа к реестрам утраченных или аннулированных паспортов нет; полную юридическую действительность по OCR установить нельзя.")

    h("2. Анализ ошибок и расхождений с разметкой")
    p("Для ФИО используется согласованность визуальной зоны и MRZ: наличие имени в обучающем словаре не требуется. Это позволяет принимать редкие имена при достаточном подтверждении распознавания.")
    p("На train принято 630 документов; пять не совпали с разметкой: tr_01506 - КАБЛЯХБАР / КАБЛЯЖБАР; tr_01665 - ДЖАМБИЧИ / ДЖАМБИУИ; tr_01765 - ХАСАНЗЯН / ХАОАНЗЯН; tr_02287 - МУТЯНСКИЙ / МУТЯНСЖИЙ; tr_02818 - ЕВГЕНИЙ / ЕВТЕНИЙ. <b>Скорее всего, эти пять расхождений связаны с опечатками в разметке.</b> В четырёх случаях с ФИО визуальная зона и MRZ совпадают; место рождения в MRZ отсутствует. Без фотографий гипотезу подтвердить нельзя, поэтому все пять учитываются как ошибки официального оценщика.")

    h("3. Протокол оценки")
    p("Групповая кросс-валидация: 5 фолдов, 3 повтора с фиксированными seed. Все снимки одного паспорта относятся к одному фолду; источники сбалансированы между фолдами. Словари органов выдачи обучаются только на обучающей части. Каждый документ получает одно проверочное предсказание в каждом повторе. Пороги зафиксированы в policy.json.")
    p("Это CV фиксированной политики, без вложенного подбора порогов. Правила и пороги ранее разрабатывались на этом наборе, поэтому проверку нельзя считать полностью слепой. Повторы используют те же документы и не увеличивают число независимых наблюдений втрое.")

    story.append(PageBreak())


    p("Результаты валидации и прогноз для Test", "title")
    p("Метрики на внешних частях CV и статистический прогноз · 25.10.2026", "sub")

    h("4. Результаты локальной проверки")
    rows = [["Выборка", "Документов", "Принято", "Ошибок по GT", "Точность", "Статус"]]
    rows.append(["Train, итоговая модель", "3000", str(training["accepted"]), str(training["bad"]), f"{100*(training['accepted']-training['bad'])/training['accepted']:.1f}%", "Обучающая выборка"])
    for index, run in enumerate(runs, 1):
        tot = run.get("totals", run)
        prec = 100 * (tot["accepted"] - tot["bad"]) / tot["accepted"] if tot["accepted"] else 0
        rows.append([f"Внешняя CV, повтор {index}", "3000", str(tot["accepted"]), str(tot["bad"]), f"{prec:.1f}%", "Out-of-fold тест"])
    table = Table([[Paragraph(str(cell), styles["small"]) for cell in row] for row in rows], colWidths=[130, 65, 55, 75, 55, 127])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), pale), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBELOW", (0, 0), (-1, 0), .6, blue), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                               ("TOPPADDING", (0, 0), (-1, -1), 5)]))
    story.append(table)
    for row in rows[1:]:
        markdown.append("- " + ": ".join((row[0], f"{row[1]} документов, {row[2]} принято, {row[3]} ошибок; {row[5]}")) + "\n")
    p("Во всех трёх повторах кросс-валидации алгоритм принимает 595 / 594 / 594 документов вне обучения при неизменных 5 несовпадениях с разметкой. Несовпадения стабильно приходятся на одни и те же пять документов, описанных выше.")

    h("5. Прогноз для test")
    rate = 100 * forecast['test_accepted'] / forecast['test_documents']
    callout(f"Принято {forecast['test_accepted']} из 2000 документов ({rate:.1f}%).<br/>Прогноз ошибок среди принятых: около {forecast['rounded_error_forecast']}.")
    expected = f"{forecast['expected_errors']:.2f}".replace('.', ',')
    sources = "; ".join(f"{s}: {r['test_accepted']} test / {r['cv_accepted']} CV" for s, r in forecast['by_source'].items())
    p(f"Распределение принятых по источникам: {sources}. Прогноз рассчитан по модели Beta-биномиального сглаживания Джеффриса Beta(0,5; 0,5) отдельно для каждого источника снимков на основе первичного повтора CV. Расчётное матожидание несовпадений составляет <b>{expected}</b>, округлённо <b>{forecast['rounded_error_forecast']}</b> (условный 95%-й предиктивный интервал: [0; {forecast['model_based_95_percent_predictive_interval'][1]}]).")
    p("Прогноз предполагает переносимость характеристик источников и независимость документов. Разметка test недоступна; фактическое число ошибок неизвестно. Целевой нулевой уровень ошибок на локальной проверке не достигнут.")

    h("6. Воспроизводимость и запуск")
    p("Решение полностью детерминировано, протестировано 39 юнит-тестами и работает офлайн на стандартном CPU без внешних библиотек.")
    p("python postprocess.py data/test_predictions.jsonl answers.jsonl<br/>python run_cv.py --folds 5 --repeats 3<br/>python -m unittest discover -s . -p \"test_*.py\"", "code")
    p("Нормативная база: Постановление Правительства РФ № 2267 (п. 24, замена паспорта в 20/45 лет); Приказ МВД РФ № 773 (приложение 24, формат MRZ). Справочник кодов: DaData (CC BY-SA 4.0).", "small")

    output = ROOT / "output/pdf"
    output.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output / "report.pdf"), pagesize=A4, rightMargin=44, leftMargin=44,
                            topMargin=36, bottomMargin=34, title="Постобработка OCR паспорта РФ", author="")
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Arial", 8)
        canvas.setFillColor(muted)
        canvas.drawString(44, 20, "Тестовое задание · Постобработка OCR паспорта РФ")
        canvas.drawRightString(A4[0] - 44, 20, f"Страница {document.page} из 2")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "REPORT.md").write_text("\n".join(markdown), encoding="utf-8", newline="\n")
    print(str(output / "report.pdf"))


if __name__ == "__main__":
    main()
