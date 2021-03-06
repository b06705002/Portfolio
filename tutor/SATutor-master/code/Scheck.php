<?php
session_start();

include("connect.php");

$table_text = "";
$chk = $_POST['chk'];
if(isset($_SESSION['teachername']))
{
  $uid = $_SESSION['username'];
  $sql = "SELECT SID FROM student_table WHERE UID = '$uid'";
  $result = $mysqli->query($sql);
  $row = $result->fetch_row();
  $date = date("Y/m/d");
  $t = date("H:i:s");
  $sql = "INSERT INTO click_table (SID,TID,Date,Period) values ('$row[0]','$chk','$date','$t')";
  if($result = $mysqli->query($sql))
  {
    $sql = "SELECT Seniority, Education,UID FROM teacher_table WHERE TID = '$chk'";
    $result = $mysqli->query($sql);
    $row = $result->fetch_row();
    $seniority = $row[0];
    $edu = $row[1];
    $sql = "SELECT Username,Sex,Age,Mail FROM member_table WHERE UID='$row[2]'";
    $result = $mysqli->query($sql);
    $row = $result->fetch_row();

    $table_text .= "<tr>";
    $table_text .= "<td>$seniority</td>";
    $table_text .= "<td>$edu</td>";
    $table_text .= "<td>$row[0]</td>";
    $table_text .= "<td>$row[1]</td>";
    $table_text .= "<td>$row[2]</td>";
    $table_text .= "<td>$row[3]</td>";
    $table_text .= "</tr>";

    $result -> free_result();
  }
  else
  {
    echo $mysqli->error;
  }

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
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Student Check</title>
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

          <table class='table table-striped bgc-transparent text-color'>
            <thead>
              <tr>
                <th>Teacher's seniority</th>
                <th>Education</th>
                <th>Name</th>
                <th>Sex</th>
                <th>Age</th>
                <th>Mail</th>
              </tr>
            </thead>
            <tbody>
              <?=$table_text?>
            </tbody>
          </table>

        </div>

        <div class="row justify-content-center">
          <a href="Sfind.php" type="button" class="btn btn-outline-light m-1">Go Back</a>
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