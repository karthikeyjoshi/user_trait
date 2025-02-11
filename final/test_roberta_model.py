from roberta_investment_traits import predict

# Test Cases
if __name__ == "__main__":
    print("\nEnter your investment-related query (or type 'exit' to quit):")
    while True:
        user_input = input("\n> ")
        if user_input.lower() == 'exit':
            break
        result = predict(user_input)
        print("\nPredicted Traits:")
        for key, value in result.items():
            print(f"{key}: {value}")
