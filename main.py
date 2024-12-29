from __func import *

def main() -> None:
    """
    Main function to load data, calculate transaction details, and generate output.
    """
    output_file = 'data/output.csv'
    stock_file = 'data/stock_prices.csv'
    purchase_file = 'data/user_purchases.csv'

    stock_data, purchase_data = load_data(stock_file, purchase_file)
    interpolated_prices = interpolate_prices(stock_data, purchase_data['time'])
    detailed_data = calculate_transaction_details(purchase_data, interpolated_prices)
    save_results_to_csv(detailed_data, output_file)
    plot_stock_with_transactions(stock_data, detailed_data)

if __name__ == '__main__':
    main()