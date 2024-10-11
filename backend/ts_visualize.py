# import os
# import pandas as pd
# from ts.plot import draw_init_line_plot, compute_y_axis_parameters

# def visualize_results():
#     try:
#         # Check if the files exist
#         if not os.path.exists('dataset.pkl'):
#             print("dataset.pkl does not exist")
#         if not os.path.exists('data.pkl'):
#             print("data.pkl does not exist")

#         # Load the data
#         dataset = pd.read_pickle('dataset.pkl')
#         data = pd.read_pickle('data.pkl')
        
#         print("Data loaded successfully")
#         print("Dataset head:\n", dataset.head())
#         print("Filtered data head:\n", data.head())
        
#         # Load metadata
#         with open('meta.txt', 'r') as f:
#             meta = f.read().split(',')
#         date_column = meta[0]
#         value_column = meta[1:]
#         value_column = [col.strip() for col in value_column]
        
#         print(f"Date column: {date_column}")
#         print(f"Value columns: {value_column}")
        
#         ts_type = 'multi' if len(value_column) > 1 else 'single'
#         output_path = os.path.join('output/ts/', ts_type)
#         if not os.path.exists(output_path):
#             os.makedirs(output_path)
        
#         # Compute global minimum and maximum values across specified columns for plotting
#         global_min, global_max = dataset[value_column].min().min(), dataset[value_column].max().max()
#         print(f"Global min: {global_min}, Global max: {global_max}")
#         y_params = compute_y_axis_parameters(global_min, global_max)
        
#         # Plot and save the initial data visualization
#         print("Generating initial plot...")
#         init_plt = draw_init_line_plot(dataset, date_column, value_column, *y_params)
#         init_graph_path = os.path.join(output_path, "init.png")
#         init_plt.savefig(init_graph_path)
#         print(f"Initial graph saved at: {init_graph_path}")
        
#         # Plot merged data and save it
#         print("Generating folded plot...")
#         plt = draw_init_line_plot(data, date_column, value_column, *y_params)
#         folded_graph_path = os.path.join(output_path, "folded.png")
#         plt.savefig(folded_graph_path)
#         print(f"Folded graph saved at: {folded_graph_path}")
        
#         return init_graph_path, folded_graph_path

#     except Exception as e:
#         print("An error occurred while generating the visualization:", str(e))
#         raise

# if __name__ == "__main__":
#     init_graph_path, folded_graph_path = visualize_results()
#     print(f"Initial graph saved at: {init_graph_path}")
#     print(f"Folded graph saved at: {folded_graph_path}")
import os
import pandas as pd
from ts.plot import draw_init_line_plot, compute_y_axis_parameters

def visualize_results(y_min=None, y_max=None):
    try:
        if not os.path.exists('dataset.pkl'):
            print("dataset.pkl does not exist")
            return None, None
        
        if not os.path.exists('data.pkl'):
            print("data.pkl does not exist")
            return None, None

        # Load the data
        dataset = pd.read_pickle('dataset.pkl')
        data = pd.read_pickle('data.pkl')
        
        print("Data loaded successfully")
        print("Dataset head:\n", dataset.head())
        print("Filtered data head:\n", data.head())
        
        # Load metadata
        with open('meta.txt', 'r') as f:
            meta = f.read().split(',')
        date_column = meta[0]
        value_column = [col.strip() for col in meta[1:]]
        
        print(f"Date column: {date_column}")
        print(f"Value columns: {value_column}")
        
        ts_type = 'multi' if len(value_column) > 1 else 'single'
        output_path = os.path.join('output/ts/', ts_type)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Compute or use provided y-axis limits
        global_min, global_max = dataset[value_column].min().min(), dataset[value_column].max().max()
        y_min = y_min if y_min is not None else global_min
        y_max = y_max if y_max is not None else global_max
        y_params = compute_y_axis_parameters(y_min, y_max)
        
        # Plot and save the initial data visualization
        print("Generating initial plot...")
        init_plt = draw_init_line_plot(dataset, date_column, value_column, *y_params)
        init_graph_path = os.path.join(output_path, "init.png")
        init_plt.savefig(init_graph_path)
        print(f"Initial graph saved at: {init_graph_path}")
        
        # Plot merged data and save it
        print("Generating folded plot...")
        plt = draw_init_line_plot(data, date_column, value_column, *y_params)
        folded_graph_path = os.path.join(output_path, "folded.png")
        plt.savefig(folded_graph_path)
        print(f"Folded graph saved at: {folded_graph_path}")
        
        return init_graph_path, folded_graph_path

    except Exception as e:
        print("An error occurred while generating the visualization:", str(e))
        return None, None
