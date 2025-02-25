import sys
from train import train_model
from evaluate import evaluate_model
from visualize import visualize_results

def main():
    print("Select an option:")
    print("1. Train the model")
    print("2. Evaluate the model")
    print("3. Visualize the results")

    choice = input("Enter the number of your choice: ")

    if choice == '1':
        print("Training the model...")
        train_model()
    elif choice == '2':
        print("Evaluating the model...")
        evaluate_model()
    elif choice == '3':
        print("Visualizing the results...")
        visualize_results()
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
