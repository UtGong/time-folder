import React, { useState, useEffect } from 'react';

function VisualizationDisplay({ initGraphPath, foldedGraphPath, isLoading, algorithmResult, buttonClicked }) {
    const [isRowLayout, setIsRowLayout] = useState(true); // Set initial state to true

    const toggleLayout = () => setIsRowLayout(!isRowLayout);
    const layoutClass = isRowLayout ? 'flex flex-row justify-around items-center' : 'flex flex-col';

    // Placeholder component with pulse animation
    const LoadingPlaceholder = () => (
        <div className="animate-pulse flex flex-col items-center justify-center p-4">
            <div className="bg-gray-300 h-48 w-full rounded-md"></div>
        </div>
    );

    // Static placeholder component
    const StaticPlaceholder = () => (
        <div className="flex flex-col items-center justify-center p-4">
            <div className="bg-gray-300 h-48 w-full rounded-md"></div>
        </div>
    );

    // Function to render image, loading placeholder, or static placeholder based on state
    const renderContent = (path, altText) => {
        if (isLoading) {
            return <LoadingPlaceholder />;
        } else if (path) {
            return <img src={path} alt={altText} className="w-full h-auto" />;
        } else if (!buttonClicked) {
            return <StaticPlaceholder />;
        }
        return null;
    };

    return (
        <div className="visualization-area p-4 border-t mt-4 shadow-xl bg-white rounded-lg">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-semibold">Visualization Output</h3>
                <div className="flex items-center">
                    <label className="mr-2 text-gray-700 font-medium">Stack</label>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input 
                            type="checkbox" 
                            checked={isRowLayout} 
                            onChange={toggleLayout} 
                            className="sr-only peer" 
                        />
                        <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-violet-500"></div>
                    </label>
                    <label className="ml-2 text-gray-700 font-medium">Row</label>
                </div>
            </div>
            <div className={layoutClass}>
                <div className="mb-4 w-full">
                    <h4>Initial Graph</h4>
                    {renderContent(initGraphPath, "Initial Graph")}
                </div>
                <div className="mb-4 w-full">
                    <h4>Folded Graph</h4>
                    {renderContent(foldedGraphPath, "Folded Graph")}
                </div>
            </div>
        </div>
    );
}

export default VisualizationDisplay;
