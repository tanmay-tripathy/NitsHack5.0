import "./App.css";
import React from "react";
import Navbar from "./components/Navbar/Navbar";
import Sidebar from "./components/Sidebar/Sidebar";
import DashBoard from "./components/DashBoard/DashBoard";

function App() {
  return (
    <div>
      <Navbar />
      <div class="container-fluid" id="main">
        <div class="row row-offcanvas row-offcanvas-left">
          <Sidebar />
          <DashBoard />
        </div>
      </div>
    </div>
  );
}

export default App;
