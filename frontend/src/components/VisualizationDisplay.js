import React, { useState } from 'react';

function VisualizationDisplay({ initGraphPath, foldedGraphPath, isLoading }) {
    const [isRowLayout, setIsRowLayout] = useState(false);

    const toggleLayout = () => setIsRowLayout(!isRowLayout);
    const layoutClass = isRowLayout ? 'flex flex-row justify-around items-center' : 'flex flex-col';

    // Placeholder component with pulse animation
    const Placeholder = () => (
        <div className="animate-pulse flex flex-col items-center justify-center p-4">
            <div className="bg-gray-300 h-48 w-full rounded-md"></div>
            <div className="mt-2 bg-gray-300 h-6 w-5/6 rounded-md"></div>
        </div>
    );

    // Function to render image, placeholder, or nothing based on isLoading and path
    const renderContent = (path, altText) => {
        if (isLoading) {
            return <Placeholder />;
        } else if (path) {
            return <img src={path} alt={altText} className="w-full h-auto" />;
        }
        // If not loading and path is not provided, render nothing or a default state
        return null;
    };

    return (
        <div className="visualization-area p-4 border-t mt-4 shadow-xl bg-white rounded-lg">
            <h3 className="text-xl font-semibold mb-4">Visualization Output</h3>
            <button 
                onClick={toggleLayout} 
                className="mb-4 bg-violet-500 hover:bg-violet-700 text-white font-bold py-2 px-4 rounded"
            >
                {isRowLayout ? 'Stack Images' : 'Align Images in Row'}
            </button>
            <div className={layoutClass}>
                <div className="mb-4">
                    <h4>Initial Graph</h4>
                    {renderContent(initGraphPath, "Initial Graph")}
                </div>
                <div className="mb-4">
                    <h4>Folded Graph</h4>
                    {renderContent(foldedGraphPath, "Folded Graph")}
                </div>
            </div>
        </div>
    );
}

export default VisualizationDisplay;

