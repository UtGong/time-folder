import React from 'react';
import Logo from '../img/logo.png'
function Header() {
    return (
        <header className="bg-gradient-to-r from-violet-800 to-violet-600 text-white p-4 flex justify-between items-center shadow-md">
            <div style={{
                display: 'flex',
                alignItems: 'center'
            }}>
              <img style={{
                width: '50px',
                height: '50px'
              }} src={Logo}/>
              <h1 style={{
                marginLeft: '10px'
              }} className="text-lg font-bold">Time Folder</h1>
            </div>
            <nav>
                <a href="#" className="p-2 hover:bg-blue-700 rounded">Home</a>
                {/* Add more navigation links as needed */}
            </nav>
        </header>
    );
}

export default Header;
