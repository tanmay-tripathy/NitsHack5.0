import React from "react";
import "../DashBoard/DashBoard.css";
import bin_f from "../../assets/bin/garbage_full.png";
import bin_h from "../../assets/bin/garbage_half_full.png";
import bin_e from "../../assets/bin/garbage-empty.png";

const DashBoard = () => {
  return (
    <div class="col main pt-5 mt-3">
      <p class="lead ">DashBoard</p>

      <div class="row mb-3">
        <div class="col-xl-3 col-sm-6 py-2">
          <div class="card bg-success text-white h-100">
            <div
              class="card-body bg-success"
              style={{ backgroundColor: "#57b960" }}
            >
              <h1 class="display-4">0</h1>
              <img className="image-card-bin" src={bin_e}></img>
              <h6 class="text-uppercase">Empty Bins</h6>
            </div>
          </div>
        </div>
        <div class="col-xl-3 col-sm-6 py-2">
          <div class="card text-white bg-danger h-100">
            <div class="card-body bg-danger">
              <h1 class="display-4">0</h1>
              <img className="image-card-bin" src={bin_h}></img>
              <h6 class="text-uppercase">Half Bins</h6>
            </div>
          </div>
        </div>
        <div class="col-xl-3 col-sm-6 py-2">
          <div class="card text-white bg-info h-100">
            <div class="card-body bg-info">
              <h1 class="display-4">0</h1>
              <img className="image-card-bin" src={bin_f}></img>
              <h6 class="text-uppercase">full Bins</h6>
            </div>
          </div>
        </div>
      </div>

      <div class="user-row">
        <div class="col-lg-10 col-md-6 col-sm-12">
          <h5 class="mt-3 mb-3 text-secondary">
            Check More Records of Each bins
          </h5>
          <div class="table-responsive">
            <table class="table table-striped">
              <thead class="thead-light">
                <tr>
                  <th>Id</th>
                  <th>Name</th>
                  <th>Location</th>
                  <th>weight</th>
                  <th>Level</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>1</td>
                  <td>D1</td>
                  <td>T point</td>
                  <td>28 kg</td>
                  <td>Medium</td>
                </tr>
              </tbody>
              <tbody>
                <tr>
                  <td>2</td>
                  <td>D5</td>
                  <td>T point</td>
                  <td>21 kg</td>
                  <td>High</td>
                </tr>
              </tbody>
              <tbody>
                <tr>
                  <td>3</td>
                  <td>D9</td>
                  <td>h9 A</td>
                  <td>11 kg</td>
                  <td>Low</td>
                </tr>
              </tbody>
              <tbody>
                <tr>
                  <td>1</td>
                  <td>D1</td>
                  <td>T point</td>
                  <td>28 kg</td>
                  <td>Medium</td>
                </tr>
              </tbody>
              <tbody>
                <tr>
                  <td>1</td>
                  <td>D1</td>
                  <td>T point</td>
                  <td>28 kg</td>
                  <td>Medium</td>
                </tr>
              </tbody>
              <tbody>
                <tr>
                  <td>1</td>
                  <td>D1</td>
                  <td>T point</td>
                  <td>28 kg</td>
                  <td>Medium</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="col-lg-8 col-md-6 col-sm-12 col-sm-offset-5">
        <h4 className="title mt-3 mb-3 text-center text-secondary">
          Data in Chart
        </h4>
        <div className="">{/* <PieChart />{" "} */}</div>
      </div>
    </div>
  );
};

export default DashBoard;
