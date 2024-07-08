import React from 'react';

function AlgorithmResults({ algorithmLoading, algorithmResult }) {
    if (algorithmLoading) {
        return (
            <div className="algorithm-results p-4 bg-gray-100 rounded-lg shadow-inner mb-4 animate-pulse">
                <h4 className="text-lg font-semibold mb-2">Algorithm Results:</h4>
                <p>Loading...</p>
            </div>
        );
    }

    if (!algorithmResult) return null;

    return (
        <div className="algorithm-results p-4 bg-gray-100 rounded-lg shadow-inner mb-4">
            <h4 className="text-lg font-semibold mb-2">Algorithm Results:</h4>
            <p>Runtime: {algorithmResult.runtime} seconds</p>
            <p>Original Timeline Length: {algorithmResult.original_length}</p>
            <p>Folded Timeline Length: {algorithmResult.folded_length}</p>
            <p>Iteration Time: 1 (Default)</p>
        </div>
    );
}

export default AlgorithmResults;
