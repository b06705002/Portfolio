<!doctype html>
<?php
  echo $_SESSION['username'];
  if(isset($_SESSION['username']))
  {
    echo '<meta http-equiv=REFRESH CONTENT=1;url=afterlogin.php>';
  }
?>
<html lang="zh-TW">
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
            <a class="nav-link active text-color" href="./index.php">Home <span class="sr-only">(current)</span></a>
          </li>
          <li class="nav-item">
            <a class="nav-link text-color" href="afterlogin.php">Dashboard</a>
          </li>
        </ul>

        <button data-toggle="modal" data-target="#login" href="login.php" type="button" class="btn btn-outline-light mt-2 m-sm-3 btn-lg">Login</button>
      </div>
    </nav>

    <div class="jumbotron jumbotron-fluid">
      <div class="container text-center push-down">
        <div class="title">
          <h1 class="heading">Tutor</h1>
        </div>
        <h5><kbd>team eight</kbd></h5>
        <p class="lead my-5">台大學生設計的家教媒合網站</p>

        <div class="row justify-content-center">
          <button data-toggle="modal" data-target="#regU" type="button" class="btn btn-outline-light m-1">Register User</button>
          <button data-toggle="modal" data-target="#regS" type="button" class="btn btn-outline-light m-1">Register Student</button>
          <button data-toggle="modal" data-target="#regT" type="button" class="btn btn-outline-light m-1">Register Teacher</button>
          <button data-toggle="modal" data-target="#login" type="button" class="btn btn-outline-light m-1">Login</button>
        </div>
      </div>
    </div>


<!-- Modal -->
<form action="register_db.php" method="post">
  <div class="modal fade" id="regU" tabindex="-1" role="dialog" aria-labelledby="m1" aria-hidden="true">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="m1">Register User</h5>
          <button type="button" class="close" data-dismiss="modal" aria-label="Close">
            <span aria-hidden="true">&times;</span>
          </button>
        </div>
        <div class="modal-body">     
          <div class="form-group">
            <label for="account">Account</label>
            <input name="UID" type="text" class="form-control" id="account" required>
          </div>
          <div class="form-group">
            <label for="username">Username</label>
            <input name="Username" type="text" class="form-control" id="username" required>
          </div>
          <div class="form-group">
            <label for="sex">Sex</label>
            <select name="Sex" class="form-control" id="sex">
              <option value="" selected disabled>Please select</option>
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </div>
          <div class="form-group">
            <label for="age">Age</label>
            <input name="Age" type="number" class="form-control" id="age" required>
          </div>
          <div class="form-group">
            <label for="mail">E-mail</label>
            <input name="Mail" type="email" class="form-control" id="mail" aria-describedby="emailHelp" required>
          </div>
          <div class="form-group">
            <label for="password">Password</label>
            <input name="Password" type="password" class="form-control" id="password" required>
            <small class="form-text text-muted">By using the Services, you signify your consent to this <a target="_blank" href="policy.html">Online Privacy Policy</a> and this site’s Terms of Use.</small>
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

<form action="student_db.php" method="post">
  <div class="modal fade" id="regS" tabindex="-1" role="dialog" aria-labelledby="m2" aria-hidden="true">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="m2">Register Student</h5>
          <button type="button" class="close" data-dismiss="modal" aria-label="Close">
            <span aria-hidden="true">&times;</span>
          </button>
        </div>
        <div class="modal-body">     
          <div class="form-group">
            <label for="accountS">Account</label>
            <input name="UID" type="text" class="form-control" id="accountS" required>
          </div>
          <div class="form-group">
            <label for="schoolS">School</label>
            <input name="school" type="text" class="form-control" id="schoolS" required>
          </div>
          <div class="form-group">
            <label for="passwordS">Password</label>
            <input name="Password" type="password" class="form-control" id="passwordS" required>
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

<form action="teacher_db.php" method="post">
  <div class="modal fade" id="regT" tabindex="-1" role="dialog" aria-labelledby="m3" aria-hidden="true">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="m3">Register Teacher</h5>
          <button type="button" class="close" data-dismiss="modal" aria-label="Close">
            <span aria-hidden="true">&times;</span>
          </button>
        </div>
        <div class="modal-body">     
          <div class="form-group">
            <label for="accountT">Account</label>
            <input name="UID" type="text" class="form-control" id="accountT" required>
          </div>
          <div class="form-group">
            <label for="educationT">Education</label>
            <input name="Education" type="text" class="form-control" id="educationT" required>
          </div>
          <div class="form-group">
            <label for="seniority">Seniority</label>
            <input name="Seniority" type="number" class="form-control" id="seniority" required>
          </div>
          <div class="form-group">
            <label for="passwordT">Password</label>
            <input name="Password" type="password" class="form-control" id="passwordT" required>
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

<form action="login_db.php" method="post">
  <div class="modal fade" id="login" tabindex="-1" role="dialog" aria-labelledby="m4" aria-hidden="true">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="m4">Login</h5>
          <button type="button" class="close" data-dismiss="modal" aria-label="Close">
            <span aria-hidden="true">&times;</span>
          </button>
        </div>
        <div class="modal-body">     
          <div class="form-group">
            <label for="accountL">Account</label>
            <input name="UID" type="text" class="form-control" id="accountL" required>
          </div>
          <div class="form-group">
            <label for="passwordL">Password</label>
            <input name="password" type="password" class="form-control" id="passwordL" required>
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


    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js" integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" crossorigin="anonymous"></script>
  </body>
</html>