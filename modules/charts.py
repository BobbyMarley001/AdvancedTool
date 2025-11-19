from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtChart import QChart, QChartView, QLineSeries
from PyQt5.QtCore import Qt

class ChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.chart = QChart()
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(Qt.AA_EnableHighDpiScaling)
        self.chart_view.setMinimumSize(400, 300)
        layout = QVBoxLayout(self)
        layout.addWidget(self.chart_view)

    def update_charts(self, logs):
        if not logs:
            return
        series = QLineSeries()
        for i, log in enumerate(logs):
            series.append(i, len(str(log['action'])))
        self.chart.removeAllSeries()
        self.chart.addSeries(series)
        self.chart.createDefaultAxes()
        self.chart.setTitle("آمار لاگ‌ها")