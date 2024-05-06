import React from 'react';

function Header() {
    return (
        <header className="bg-gradient-to-r from-violet-800 to-violet-600 text-white p-4 flex justify-between items-center shadow-md">
            <h1 className="text-lg font-bold">Time Folder</h1>
            <nav>
                <a href="#" className="p-2 hover:bg-blue-700 rounded">Home</a>
                {/* Add more navigation links as needed */}
            </nav>
        </header>
    );
}

export default Header;
