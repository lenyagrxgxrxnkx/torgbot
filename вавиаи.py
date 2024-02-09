import sys
import os
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QListWidget, \
    QHBoxLayout, QMessageBox


class StockPricePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Price Predictor")
        self.setGeometry(100, 100, 800, 600)

        self.portfolio_file = 'portfolio.txt'

        self.init_ui()
        self.load_portfolio()

    def init_ui(self):
        layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        self.symbol_label = QLabel("Enter the stock symbol:")
        input_layout.addWidget(self.symbol_label)
        self.symbol_input = QLineEdit()
        input_layout.addWidget(self.symbol_input)
        self.add_button = QPushButton("Add to Portfolio")
        self.add_button.clicked.connect(self.add_to_portfolio)
        input_layout.addWidget(self.add_button)
        layout.addLayout(input_layout)

        self.portfolio_list = QListWidget()
        self.portfolio_list.itemClicked.connect(self.show_portfolio_stock_graph)
        layout.addWidget(self.portfolio_list)

        button_layout = QHBoxLayout()
        self.remove_button = QPushButton("Remove from Portfolio")
        self.remove_button.clicked.connect(self.remove_from_portfolio)
        button_layout.addWidget(self.remove_button)

        self.predict_button = QPushButton("Predict Price")
        self.predict_button.clicked.connect(self.predict_price)
        button_layout.addWidget(self.predict_button)

        layout.addLayout(button_layout)

        self.output_label = QLabel("")
        layout.addWidget(self.output_label)

        self.setLayout(layout)

    def get_stock_data(self, symbol, start_date, end_date):
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data['Adj Close']

    def train_linear_regression(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    def predict_next_price(self, model, latest_price):
        return model.predict([[latest_price]])[0]

    def plot_stock_price(self, symbol, prices, current_price, predicted_price):
        plt.figure(figsize=(10, 6))
        plt.plot(prices, label='Actual Price')
        if current_price > predicted_price:
            plt.axhline(y=current_price, color='g', linestyle='--', label='Current Price')
            plt.axhline(y=predicted_price, color='r', linestyle='--', label='Predicted Price')
        else:
            plt.axhline(y=current_price, color='r', linestyle='--', label='Current Price')
            plt.axhline(y=predicted_price, color='g', linestyle='--', label='Predicted Price')
        plt.title(f"Stock Price of {symbol}")
        plt.xlabel("Days")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_portfolio_stock_graph(self, item):
        symbol = item.text()

        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

        stock_prices = self.get_stock_data(symbol, start_date, end_date)
        current_price = stock_prices[-1]

        X = np.array(range(1, len(stock_prices) + 1)).reshape(-1, 1)
        y = np.array(stock_prices)

        model = self.train_linear_regression(X, y)

        predicted_price = self.predict_next_price(model, current_price)

        self.plot_stock_price(symbol, stock_prices, current_price, predicted_price)

    def predict_price(self):
        symbol = self.symbol_input.text().upper()


        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

        stock_prices = self.get_stock_data(symbol, start_date, end_date)

        X = np.array(range(1, len(stock_prices) + 1)).reshape(-1, 1)
        y = np.array(stock_prices)

        model = self.train_linear_regression(X, y)

        latest_price = stock_prices[-1]
        next_price = self.predict_next_price(model, latest_price)

        current_price = latest_price

        self.output_label.setText(f"Current price for {symbol}: ${current_price:.2f}\n"
                                   f"Predicted next price for {symbol}: ${next_price:.2f}")

        self.plot_stock_price(symbol, stock_prices, current_price, next_price)

    def add_to_portfolio(self):
        symbol = self.symbol_input.text().upper()
        if not symbol:
            return

        if symbol not in [self.portfolio_list.item(i).text() for i in range(self.portfolio_list.count())]:
            self.portfolio_list.addItem(symbol)
            with open(self.portfolio_file, 'a') as file:
                file.write(symbol + '\n')
        else:
            QMessageBox.warning(self, 'Duplicate Symbol', 'This symbol is already in the portfolio.')

    def remove_from_portfolio(self):
        selected_items = self.portfolio_list.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            self.portfolio_list.takeItem(self.portfolio_list.row(item))

        with open(self.portfolio_file, 'w') as file:
            for i in range(self.portfolio_list.count()):
                file.write(self.portfolio_list.item(i).text() + '\n')

    def load_portfolio(self):
        if not os.path.exists(self.portfolio_file):
            return

        with open(self.portfolio_file, 'r') as file:
            symbols = file.readlines()
            for symbol in symbols:
                self.portfolio_list.addItem(symbol.strip())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPricePredictor()
    window.show()
    sys.exit(app.exec_())
