import matplotlib.pyplot as plt

# File containing the float data
file_name = "pendulum_positions.txt"

def plot_floats(file_name):
    try:
        # Read the float data from the file
        with open(file_name, "r") as file:
            data = [float(line.strip()) for line in file.readlines()]
        
        # Generate indices for the x-axis
        indices = list(range(len(data)))
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(indices, data, marker='o', linestyle='-', color='b', label="Pendulum Positions")
        plt.title("Pendulum Joint Positions")
        plt.xlabel("Index")
        plt.ylabel("Position Value")
        plt.grid(True)
        plt.legend()
        
        # Show the plot
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    plot_floats(file_name)
