import React from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import MainArea from './components/MainArea';

function App() {
  const columns = ['Column 1', 'Column 2', 'Column 3', 'Column 4'];

  return (
    <div className="flex flex-col min-h-full">
      <Header />
      <main className="flex-grow">
        <MainArea columns={columns} />
      </main>
      <Footer />
    </div>
  );
}

export default App;
