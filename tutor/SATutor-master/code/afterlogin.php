<!doctype html>
<html lang="en">
<head>
  <!-- Required meta tags -->
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

  <!-- Bootstrap CSS -->
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">

  <title>Tutor</title>
  <link rel="stylesheet" href="./style.css">
</head>
<body>
  <nav class="navbar navbar-expand-sm navbar-dark px-5">
    <a class="navbar-brand text-color" href="#">Tutor</a>
    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon text-color"></span>
    </button>

    <div class="collapse navbar-collapse" id="navbarSupportedContent">
      <ul class="navbar-nav mr-auto">
        <li class="nav-item">
          <a class="nav-link text-color" href="./index.php">Home</a>
        </li>
        <li class="nav-item">
          <a class="nav-link active text-color" href="afterlogin.php">Dashboard <span class="sr-only">(current)</span></a>
        </li>
      </ul>

      <a href="logout_db.php" type="button" class="btn btn-outline-light mt-2 m-sm-3 btn-lg">logout</a>
    </div>
  </nav>

  <div class="jumbotron jumbotron-fluid">
    <div class="container text-center push-down">
      <div class="title">
        <h1 class="heading">Tutor</h1>
      </div>
      <h5><kbd>team eight</kbd></h5>
      <!-- <p class="lead my-5">台大學生設計的家教媒合網站</p> -->

      <div class="content my-5 py-3 p-sm-4">
        <div class="row justify-content-center my-3">
          <div class="col-sm-6">
            <p class="lead sub-title">Post a request</p>
          </div>
          <div class="col-sm-6 sub-btn">
            <button data-toggle="modal" data-target="#dbmain" type="button" class="btn btn-outline-light m-1">Enter post web</button>
          </div>
        </div>

        <div class="dropdown-divider d-block d-sm-none"></div>

        <div class="row justify-content-center my-3">
          <div class="col-sm-6">
            <p class="lead sub-title">Choose your identity</p>
          </div>
          <div class="col-sm-6 sub-btn">
            <a href="Tfind.php" type="button" class="btn btn-outline-light m-1">Teacher</a>
            <a href="Sfind.php" type="button" class="btn btn-outline-light m-1">Student</a>
          </div>
        </div>
      </div>
    </div>
  </div>

  <form action="DBmaindb.php" method="post">
    <div class="modal fade" id="dbmain" tabindex="-1" role="dialog" aria-labelledby="m1" aria-hidden="true">
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="m1">Post a request</h5>
            <button type="button" class="close" data-dismiss="modal" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>
          <div class="modal-body">     
            <div class="form-group">
              <label for="region">Choose your studying/teaching region</label>
              <select name="Region" class="form-control" id="region" required>
                <option value="" selected disabled>Please select</option>
                <option value="Taipei">Taipei</option>
                <option value="New Taipei">New Taipei</option>
                <option value="Keelung">Keelung</option>
                <option value="Tainan">Tainan</option>
                <option value="Hsinchu">Hsinchu</option>
                <option value="Chiayi">Chiayi</option>
              </select>
            </div>
            <div class="form-group">
              <label for="Subject">Choose Subject</label>
              <select name="Subject" class="form-control" id="Subject" required>
                <option value="" selected disabled>Please select</option>
                <option value="Chinese">Chinese</option>
                <option value="English">English</option>
                <option value="Math">Math</option>
                <option value="Programming">Programming</option>
                <option value="History">History</option>
                <option value="Sciense">Sciense</option>
              </select>
            </div>
            <div class="form-group">
              <label for="time">Choose your studying/teaching time</label>
              <input name="Time" type="text" class="form-control" id="time" required>
            </div>
            <div class="form-group">
              <label for="st">Selcet S/T if you are a student/teacher</label>
              <select name="ST" class="form-control" id="st">
                <option value="" selected disabled>Please select</option>
                <option value="S">S</option>
                <option value="T">T</option>
              </select>
            </div>
            <div class="form-group">
              <label for="seniority">Enter you seniority</label>
              <textarea name="Seniority" type="text" class="form-control" id="seniority" placeholder="Chinese: 1 year..." required></textarea>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
            <button type="submit" class="btn btn-primary">Submit</button>
          </div>
        </div>
      </div>
    </div>
  </form>




<script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
  <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js" integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" crossorigin="anonymous"></script>
</body>
</html>