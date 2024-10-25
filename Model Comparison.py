# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:17:40 2024

@author: pahar
"""


def setup_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            # Enable dynamic memory growth for GPU to avoid allocating all GPU memory at once
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Using GPU: {physical_devices[0]}")
        except RuntimeError as e:
            print(f"Error while setting up GPU: {e}")
    else:
        print("GPU is not available, using CPU.")

setup_gpu()
def run_optimization_comparison(X_train1, y_train1, X_val1, y_val1, X_test1, y_test1, timesteps=4, features=1):
    # Define bounds for parameters
    bounds = {
        'lstm_units': (2, 10),
        'dense_units': (4, 16),
        'learning_rate': (0.0001, 0.01)
    }

    # 1. Run DBO Optimizer
    print("\n=== Running DBO Optimizer ===")
    dbo_optimizer = DBOLSTMOptimizer(n_beetles=10, n_iterations=20, bounds=bounds)
    best_dbo_params, best_dbo_fitness = dbo_optimizer.optimize(
        X_train1, y_train1, X_val1, y_val1, timesteps, features
    )

    # Create and evaluate best DBO model
    best_dbo_model = create_base_lstm_model(
        timesteps=timesteps,
        features=features,
        lstm_units=best_dbo_params['lstm_units'],
        dense_units=best_dbo_params['dense_units'],
        learning_rate=best_dbo_params['learning_rate']
    )
    best_dbo_model.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=100, batch_size=32, verbose=0)
    dbo_predictions = best_dbo_model.predict(X_test1)
    dbo_rmse = sqrt(mean_squared_error(y_test1, dbo_predictions))

    # 2. Run Genetic Algorithm Optimizer
    print("\n=== Running Genetic Algorithm Optimizer ===")
    ga_optimizer = GALSTMOptimizer(population_size=20, n_generations=20, bounds=bounds)
    best_ga_params, best_ga_fitness = ga_optimizer.optimize(
        X_train1, y_train1, X_val1, y_val1, timesteps, features
    )

    # Create and evaluate best GA model
    best_ga_model = create_base_lstm_model(
        timesteps=timesteps,
        features=features,
        lstm_units=best_ga_params['lstm_units'],
        dense_units=best_ga_params['dense_units'],
        learning_rate=best_ga_params['learning_rate']
    )
    best_ga_model.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=100, batch_size=32, verbose=0)
    ga_predictions = best_ga_model.predict(X_test1)
    ga_rmse = sqrt(mean_squared_error(y_test1, ga_predictions))

    # 3. Run PSO Optimizer
    print("\n=== Running PSO Optimizer ===")
    pso_optimizer = PSOLSTMOptimizer(population_size=20, n_iterations=20, bounds=bounds)
    best_pso_params, best_pso_fitness = pso_optimizer.optimize(
        X_train1, y_train1, X_val1, y_val1, timesteps, features
    )

    # Create and evaluate best PSO model
    best_pso_model = create_base_lstm_model(
        timesteps=timesteps,
        features=features,
        lstm_units=best_pso_params['lstm_units'],
        dense_units=best_pso_params['dense_units'],
        learning_rate=best_pso_params['learning_rate']
    )
    best_pso_model.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=100, batch_size=32, verbose=0)
    pso_predictions = best_pso_model.predict(X_test1)
    pso_rmse = sqrt(mean_squared_error(y_test1, pso_predictions))

    # 4. Run GSA Optimizer
    print("\n=== Running GSA Optimizer ===")
    gsa_optimizer = GSALSTMOptimizer(population_size=20, n_iterations=20, bounds=bounds)
    best_gsa_params, best_gsa_fitness = gsa_optimizer.optimize(
        X_train1, y_train1, X_val1, y_val1, timesteps, features
    )

    # Create and evaluate best GSA model
    best_gsa_model = create_base_lstm_model(
        timesteps=timesteps,
        features=features,
        lstm_units=best_gsa_params['lstm_units'],
        dense_units=best_gsa_params['dense_units'],
        learning_rate=best_gsa_params['learning_rate']
    )
    best_gsa_model.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=100, batch_size=32, verbose=0)
    gsa_predictions = best_gsa_model.predict(X_test1)
    gsa_rmse = sqrt(mean_squared_error(y_test1, gsa_predictions))

    # 5. Run RDA Optimizer
    print("\n=== Running RDA Optimizer ===")
    rda_optimizer = RDALSTMOptimizer(population_size=20, n_iterations=20, bounds=bounds)
    best_rda_params, best_rda_fitness = rda_optimizer.optimize(
        X_train1, y_train1, X_val1, y_val1, timesteps, features
    )

    # Create and evaluate best RDA model
    best_rda_model = create_base_lstm_model(
        timesteps=timesteps,
        features=features,
        lstm_units=best_rda_params['lstm_units'],
        dense_units=best_rda_params['dense_units'],
        learning_rate=best_rda_params['learning_rate']
    )
    best_rda_model.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=100, batch_size=32, verbose=0)
    rda_predictions = best_rda_model.predict(X_test1)
    rda_rmse = sqrt(mean_squared_error(y_test1, rda_predictions))

    # Print results
    print("\n=== Optimization Results ===")
    print("\nDBO LSTM Results:")
    print(f"Best parameters: {best_dbo_params}")
    print(f"Test RMSE: {dbo_rmse}")

    print("\nGenetic Algorithm LSTM Results:")
    print(f"Best parameters: {best_ga_params}")
    print(f"Test RMSE: {ga_rmse}")

    print("\nPSO LSTM Results:")
    print(f"Best parameters: {best_pso_params}")
    print(f"Test RMSE: {pso_rmse}")

    print("\nGSA LSTM Results:")
    print(f"Best parameters: {best_gsa_params}")
    print(f"Test RMSE: {gsa_rmse}")

    print("\nRDA LSTM Results:")
    print(f"Best parameters: {best_rda_params}")
    print(f"Test RMSE: {rda_rmse}")

    return {
        'dbo': {
            'model': best_dbo_model,
            'params': best_dbo_params,
            'rmse': dbo_rmse,
            'predictions': dbo_predictions
        },
        'ga': {
            'model': best_ga_model,
            'params': best_ga_params,
            'rmse': ga_rmse,
            'predictions': ga_predictions
        },
        'pso': {
            'model': best_pso_model,
            'params': best_pso_params,
            'rmse': pso_rmse,
            'predictions': pso_predictions
        },
        'gsa': {
            'model': best_gsa_model,
            'params': best_gsa_params,
            'rmse': gsa_rmse,
            'predictions': gsa_predictions
        },
        'rda': {
            'model': best_rda_model,
            'params': best_rda_params,
            'rmse': rda_rmse,
            'predictions': rda_predictions
        }
    }

# Function to plot results
def plot_results(results, y_test1):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 6))
    plt.plot(y_test1, label='Actual', color='black')
    plt.plot(results['dbo']['predictions'], label='DBO-LSTM', alpha=0.7)
    plt.plot(results['ga']['predictions'], label='GA-LSTM', alpha=0.7)
    plt.plot(results['pso']['predictions'], label='PSO-LSTM', alpha=0.7)
    plt.plot(results['gsa']['predictions'], label='GSA-LSTM', alpha=0.7)
    plt.plot(results['rda']['predictions'], label='RDA-LSTM', alpha=0.7)
    plt.legend()
    plt.title('Comparison of Different Optimization Approaches')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# Function to compare RMSE values
def compare_rmse(results):
    import matplotlib.pyplot as plt

    algorithms = list(results.keys())
    rmse_values = [results[algo]['rmse'] for algo in algorithms]

    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, rmse_values)
    plt.title('RMSE Comparison Across Different Optimization Algorithms')
    plt.ylabel('RMSE')
    plt.show()

if __name__ == "__main__":
    # Assuming X_train1, y_train1, etc. are already defined as in your code
    results = run_optimization_comparison(X_train1, y_train1, X_val1, y_val1, X_test1, y_test1)
    plot_results(results, y_test1)

"""## Extra View"""



# Example Usage
bounds = {
    'lstm_units': [10, 100],
    'dense_units': [10, 50],
    'learning_rate': [0.0001, 0.01]
}

optimizer = HybridOptimizer(population_size=30, n_generations=50, bounds=bounds)
best_solution, best_fitness = optimizer.optimize()
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)