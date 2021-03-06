<?php
session_start();
include("connect.php");
$acp = $_POST['acp']; //TCID
$sid = $_SESSION['studentname'];
$sql = "SELECT * FROM T_case_table WHERE TCID='$acp'";// TCID, TID, Region, Subject, Time
$result0 = $mysqli->query($sql);
$row = $result0->fetch_row();
$tid = $row[1];

$sql = "INSERT INTO case_table (SID, TID, Time, Region, Subject) values ('$sid','$tid','$row[4]','$row[2]','$row[3]')";
$result2 = $mysqli->query($sql);

$sql = "DELETE FROM T_case_table WHERE TCID='$acp'";
$result3 = $mysqli->query($sql);
$text = "";
if($result0 && $result2)
{
  $text .= "<div>&nbsp</div> <div>&nbsp</div><div>&nbsp</div> <div class='container'><div class='alert alert-primary'><strong>Deal!!!</strong> succeed to accept. </div></div>";
}
else
{
  $text .= "<div>&nbsp</div> <div>&nbsp</div><div>&nbsp</div> <div class='container'><div class='alert alert-danger'><strong>Something wrong!!!</strong> fail to accept. </div></div>";
}
?>
<!doctype html>
<html lang="zh-TW">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">

    <title>Accept</title>
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

        <div class="row justify-content-center mt-5">
          <?=$text?>
        </div>

        <div class="row justify-content-center">
          <a href="afterlogin.php" type="button" class="btn btn-outline-light m-1">Go Back</a>
        </div>

      </div>
    </div>
    


  <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js" integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" crossorigin="anonymous"></script>
  </body>
  <footer>
    <div class="container">
    <a href = "Sfind.php">
    <div class="text-center"  class="font-weight-bold">
      <h5 class="text-white"><kbd>Go Back</kbd></h5>
    </div>
    </a>
  </div>
  </footer>
</html>