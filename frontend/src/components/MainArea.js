import React, { useState, useEffect, useRef } from "react";
import VisualizationDisplay from "./VisualizationDisplay";

function MainArea() {
  const [visualizationType, setVisualizationType] = useState("Time Series");
  const [weight, setWeight] = useState(10); // Initial weight setting
  const [weightSliderRange, setWeightSliderRange] = useState("0-20"); // Range for the weight slider
  const [columns, setColumns] = useState([]); // Columns fetched from backend
  const [selectedColumns, setSelectedColumns] = useState([]); // User-selected columns for visualization

  const [initGraphPath, setInitGraphPath] = useState(""); // Initial graph image path
  const [foldedGraphPath, setFoldedGraphPath] = useState(""); // Folded graph image path
  const [isLoading, setIsLoading] = useState(false); // Loading state to manage UI during fetch

  const visualizationRef = useRef(null);

  useEffect(() => {
    // Fetch column data based on the visualization type
    const fetchData = async () => {
      const dataPath = visualizationType === 'Time Series'
        ? 'data/StnData_2020-2023_dailytemp.csv'
        : 'data/Foreign_Exchange_Rates_Filled_Corrected.csv';

      try {
        const response = await fetch('http://127.0.0.1:5000/get-columns', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dataPath }),
        });
        if (!response.ok) throw new Error('Failed to fetch columns');
        const result = await response.json();
        setColumns(result.columns);
        setSelectedColumns([]);
      } catch (error) {
        console.error('Failed to fetch columns:', error);
      }
    };

    fetchData();
  }, [visualizationType]);

  useEffect(() => {
    // Scroll to the visualization section when images are loaded
    if (initGraphPath && foldedGraphPath && !isLoading) {
      visualizationRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [initGraphPath, foldedGraphPath, isLoading]);

  const handleSubmit = async () => {
    setIsLoading(true);
    let payload;
    switch (visualizationType) {
      case "Time Series":
        payload = {
          script_name: "ts_main",
          data_name:
            visualizationType === "Time Series"
              ? "data/StnData_2020-2023_dailytemp.csv"
              : "data/Foreign_Exchange_Rates_Filled_Corrected.csv",
          weight: weight,
          additional_params: [
            visualizationType === "Time Series" ? "2020-01-01" : "2000-01-01",
            visualizationType === "Time Series" ? "2023-12-31" : "2019-12-31",
            visualizationType === "Time Series" ? "Date" : "Time Serie",
            selectedColumns,
          ],
        };
        break;
      case "Stack Graph":
        payload = {
          script_name: "ts_main",
          data_name: "data/Foreign_Exchange_Rates_Filled_Corrected.csv",
          weight: weight,
          additional_params: [
            "2000-01-01",
            "2019-12-31",
            "Time Serie",
            selectedColumns,
          ],
        };
        break;
      case "Extended Gantt Graph":
        payload = {
          script_name: "ss_main",
          data_name: "HR",
          weight: weight,
        };
        break;
      case "Relationship Graph":
        payload = {
          script_name: "network_graph_main",
          data_name: "flights",
          weight: weight,
        };
        break;
      default:
        console.error("Invalid visualization type");
        return;
    }
    try {
      const response = await fetch("http://127.0.0.1:5000/visualize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json(); // Parsing the JSON response body
      console.log(data);

      // Assuming data contains base64 encoded images
      if (data.init_image && data.folded_image) {
        setInitGraphPath(data.init_image);
        setFoldedGraphPath(data.folded_image);
        console.log("Done encoding images");
      }
    } catch (error) {
      console.error("Error:", error);
    }
    setIsLoading(false);
  };

  const handleColumnChange = (columnName) => {
    setSelectedColumns((prevSelectedColumns) =>
      prevSelectedColumns.includes(columnName)
        ? prevSelectedColumns.filter((col) => col !== columnName)
        : [...prevSelectedColumns, columnName]
    );
  };

  const handleWeightChange = (event) => {
    setWeight(parseFloat(event.target.value));
  };

  const handleWeightSliderRangeChange = () => {
    setWeightSliderRange(weightSliderRange === "0-1" ? "0-20" : "0-1");
  };

  return (
    <div className="flex flex-col gap-4 p-4 h-full bg-gradient-to-br from-violet-100 to-violet-300">
      <div className="flex gap-4 items-center bg-white shadow-xl rounded-lg p-4">
        <div className="flex flex-col flex-1">
          <label
            htmlFor="visualization-type"
            className="mb-1 text-gray-700 text-sm font-bold"
          >
            Visualization:
          </label>
          <select
            id="visualization-type"
            value={visualizationType}
            onChange={(e) => setVisualizationType(e.target.value)}
            className="bg-gray-50 border border-gray-300 rounded shadow p-2 focus:outline-none focus:ring focus:border-violet-300"
          >
            <option value="Time Series">Time Series</option>
            <option value="Stack Graph">Stack Graph</option>
            <option value="Extended Gantt Graph">Extended Gantt Graph</option>
            <option value="Relationship Graph">Relationship Graph</option>
          </select>
        </div>
        <div className="flex flex-col flex-1">
          <label
            htmlFor="weight"
            className="mb-1 text-violet-700 text-sm font-bold"
          >
            Weight (Larger values fold more time frames):
          </label>
          <input
            type="range"
            id="weight"
            min="0"
            max={weightSliderRange === "0-1" ? "1" : "20"}
            step={weightSliderRange === "0-1" ? "0.01" : "1"}
            value={weight}
            onChange={handleWeightChange}
            className="flex-1 h-2 bg-violet-200 rounded-full appearance-none cursor-pointer transition-colors duration-200 ease-in-out"
          />
          <span className="text-violet-700 font-medium">{weight.toFixed(2)}</span>
        </div>
        <button
          onClick={handleWeightSliderRangeChange}
          className="bg-violet-200 text-violet-800 font-bold py-2 px-4 rounded"
        >
          {weightSliderRange === "0-1" ? "Switch to 0-20" : "Switch to 0-1"}
        </button>
      </div>
      {(visualizationType === "Time Series" ||
        visualizationType === "Stack Graph") && (
        <div className="flex flex-col bg-white shadow-xl rounded-lg p-4">
          <span className="text-gray-700 text-sm font-bold mb-2">
            Data Columns:
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
            {columns.map((column) => (
              <label key={column} className="inline-flex items-center">
                <input
                  type="checkbox"
                  id={column}
                  checked={selectedColumns.includes(column)}
                  onChange={() => handleColumnChange(column)}
                  className="form-checkbox text-violet-600"
                />
                <span className="ml-2 text-gray-700">{column}</span>
              </label>
            ))}
          </div>
        </div>
      )}
      <button
        onClick={handleSubmit}
        disabled={isLoading}
        className="mt-4 bg-violet-500 hover:bg-violet-700 text-white font-bold py-2 px-4 rounded inline-flex items-center justify-center w-full"
      >
        {isLoading && (
          <svg
            className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            ></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5a7.987 7.987 0 014-1.528V10H0v2c0 1.5.418 2.9 1.132 4h4.736A7.987 7.987 0 016 17zm6 5v-4a8 8 0 018 8h4c0-6.627-5.373-12-12-12v4z"
            ></path>
          </svg>
        )}
        {isLoading ? "Loading..." : "Submit"}
      </button>
      <div ref={visualizationRef}>
        <VisualizationDisplay
          initGraphPath={initGraphPath}
          foldedGraphPath={foldedGraphPath}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}

export default MainArea;
