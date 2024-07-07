import scripts.create_directories
import scripts.download_data
import scripts.prediction
import scripts.prepare_data
import scripts.train_model
import scripts.evaluate_model
import scripts.fine_tune_model
import scripts.prediction
from config.config import ticker_symbol, interval, timesteps, model_name, seasonal_period, epochs, batch_size, forecast_days


def create_directories():
    print("Creating directories...")
    scripts.create_directories.main()

def download_data():
    print("Uploading data...")
    scripts.download_data.main(ticker_symbol, interval)
    
def prepare_data():
    print("Data preparation...")
    scripts.prepare_data.main(ticker_symbol, seasonal_period)

def train_model():
    print("Model training...")
    scripts.train_model.main(ticker_symbol, timesteps, epochs, batch_size, model_name)

def evaluate_model():
    print("Evaluation of the model...")
    scripts.evaluate_model.main(ticker_symbol, timesteps, model_name)

def fine_tune_model():
    print("Fine-tuning the model...")
    scripts.fine_tune_model.main(ticker_symbol, model_name, timesteps)
    
def prediction():
    print("Predicting...")
    scripts.prediction.main(ticker_symbol, model_name, forecast_days, timesteps)

def do_all_actions():
    create_directories()
    download_data()
    prepare_data()
    train_model()
    evaluate_model()
    fine_tune_model()
    prediction()

def main_menu():
    while True:
        print("\n===== Main menu =====")
        print("1. Create directories")
        print("2. Upload data")
        print("3. Prepare data")
        print("4. Train model")
        print("5. Evaluate the model")
        print("6. Fine-tuning the model")
        print("7. Predict")
        print("8. Perform all actions")
        print("9. Exit")

        choice = input("Enter your choice (1-9): ")

        if choice == '1':
            create_directories()
        elif choice == '2':
            download_data()
        elif choice == '3':
            prepare_data()
        elif choice == '4':
            train_model()
        elif choice == '5':
            evaluate_model()
        elif choice == '6':
            fine_tune_model()
        elif choice=='7':
            prediction()
        elif choice == '8':
            do_all_actions()
        elif choice == '9':
            print("Completion of the program.")
            break
        else:
            print("Wrong choice. Please enter a number from 1 to 9.")

if __name__ == "__main__":
    main_menu()
