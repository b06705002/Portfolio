<?php

session_start();
include("connect.php");
$uid = $_SESSION['username'];
$sql = "SELECT TID FROM teacher_table WHERE UID='$uid'";
$result = $mysqli->query($sql);
$row = $result->fetch_row();

$_SESSION['teachername'] = $row[0];
$tid = $row[0];

$table1_text = "";
$table2_text = "";
$table3_text = "";


$sql = "SELECT * FROM case_table WHERE TID='$tid'"; // SID, TID, Time, Region, Subject
if ($result = $mysqli -> query($sql)) {
  $count = 0;
  while ($row = $result -> fetch_row()) {
    $count += 1;
    $tmpsid = $row[0];
    $tmpsql = "SELECT UID FROM student_table WHERE SID='$tmpsid'";
    $tmpsql = "SELECT Username FROM (select student_table.UID, school, Username, SID from student_table inner join member_table on student_table.UID = member_table.uid) as t where SID='$tmpsid'";
    $tmpres = $mysqli -> query($tmpsql);
    $tmpsuid = $tmpres -> fetch_row();
    $table1_text .= "<tr>";
    $table1_text .= "<td>$tmpsuid[0]</td>";
    $table1_text .= "<td>$row[2]</td>";
    $table1_text .= "<td>$row[3]</td>";
    $table1_text .= "<td>$row[4]</td>";
    $table1_text .= "</tr>";
  }
  $result -> free_result();
}

$sql = "SELECT * FROM click_table WHERE TID='$tid'"; // SID, TID, Date, Period
if ($result = $mysqli -> query($sql)) {
  $count = 0;
  while ($row = $result -> fetch_row()) {
    $count += 1;
    $tmpsid = $row[0];
    $tmpsql = "SELECT UID FROM student_table WHERE SID='$tmpsid'";
    $tmpsql = "SELECT Username FROM (select student_table.UID, school, Username, SID from student_table inner join member_table on student_table.UID = member_table.uid) as t where SID='$tmpsid'";
    $tmpres = $mysqli -> query($tmpsql);
    $tmpsuid = $tmpres -> fetch_row();
    $table2_text .= "<tr>";
    $table2_text .= "<td>$tmpsuid[0]</td>";
    $table2_text .= "<td>$row[2]</td>";
    $table2_text .= "<td>$row[3]</td>";
    $table2_text .= "</tr>";
  }
  $result -> free_result();
}

if ($_GET['find-region']) {
  $sql = "SELECT * FROM S_case_table WHERE Region = '".$_GET['find-region']."'";
}
elseif ($_GET['find-subject']) {
  $sql = "SELECT * FROM S_case_table WHERE Subject = '".$_GET['find-subject']."'";
}
elseif ($_GET['find-subject'] && $_GET['find-region']) {
  $sql = "SELECT * FROM S_case_table WHERE Subject = '".$_GET['find-subject']."' AND Region = '".$_GET['find-region']."'";
}
else {
  $sql = "SELECT * FROM S_case_table";
}

$result = $mysqli->query($sql);
if ($result = $mysqli -> query($sql)) {
  $count = 0;
  while ($row = $result -> fetch_row()) {
    $count += 1;
    $tmpsid = $row[1];
    $tmpsql = "SELECT UID FROM student_table WHERE SID='$tmpsid'";
    $tmpsql = "SELECT Username FROM (select student_table.UID, school, Username, SID from student_table inner join member_table on student_table.UID = member_table.uid) as t where SID='$tmpsid'";
    $tmpres = $mysqli -> query($tmpsql);
    $tmpsuid = $tmpres -> fetch_row();
    $table3_text .= "<tr>";
    $table3_text .= "<td>$tmpsuid[0]</td>";
    $table3_text .= "<td>$row[2]</td>";
    $table3_text .= "<td>$row[3]</td>";
    $table3_text .= "<td>$row[4]</td>";
    $table3_text .= "<td><pre class='text-color'>$row[5]<pre></td>";
    // $table3_text .= "<td><form action='tcheck.php' method='post'><input style='color:white' type='submit' name='chk' value=$row[1]></form></td>";
    $table3_text .= "<td><input class='checkbox chk' type='checkbox' name='chk' value=$row[1]><form action='tcheck.php' method='post'><input class='d-none' type='text' name='chk' value=$row[1]></form></td>";
    // $table3_text .= "<td><form action='Taccept.php' method='post'><input style='color:white' type='submit' name='acp' value=$row[0]></form></td>";
    $table3_text .= "<td><input class='checkbox acp' type='checkbox' name='acp' value=$row[0]><form action='Taccept.php' method='post'><input class='d-none' type='text' name='acp' value=$row[0]></form></td>";
    $table3_text .= "</tr>";
  }
  $result -> free_result();
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
    <title>Teacher Find</title>
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
      <div class="container text-center">
        <div class="title">
          <h1 class="heading">Tutor</h1>
        </div>
        <h5><kbd>team eight</kbd></h5>

        <div class="row justify-content-center">
          <div class="col-md-6">
            <p class="lead font-weight-bold">Your Case</p>

            <table class='table table-striped bgc-transparent text-color'>
              <thead>
                <tr>
                  <th>Student account</th>
                  <th>Time</th>
                  <th>Region</th>
                  <th>Subject</th>
                </tr>
              </thead>
              <tbody>
                <?=$table1_text?>
              </tbody>
            </table>

          </div>
          <div class="col-md-6">
            <p class="lead font-weight-bold">You have seen</p>

            <table class='table table-striped bgc-transparent text-color'>
              <thead>
                <tr>
                  <th>Student account</th>
                  <th>Date</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                <?=$table2_text?>
              </tbody>
            </table>

          </div>
        </div>


        <div class="row justify-content-center">
          <form class="form-inline w-100 justify-content-end my-2">
            <label class="mr-2" for="find-region">Region:</label>
            <select name="find-region" class="form-control mr-sm-2 search-input" id="find-region">
              <option value="" selected>ALL</option>
              <option value="Taipei">Taipei</option>
              <option value="New Taipei">New Taipei</option>
              <option value="Keelung">Keelung</option>
              <option value="Tainan">Tainan</option>
              <option value="Hsinchu">Hsinchu</option>
              <option value="Chiayi">Chiayi</option>
            </select>

            <label class="mr-2" for="find-subject">Subject:</label>
            <select name="find-subject" class="form-control mr-sm-2 search-input" id="find-subject">
              <option value="" selected>ALL</option>
              <option value="Chinese">Chinese</option>
              <option value="English">English</option>
              <option value="Math">Math</option>
              <option value="Programming">Programming</option>
              <option value="History">History</option>
              <option value="Sciense">Sciense</option>
            </select>
            <!-- <input type="text" class="form-control mr-sm-2 search-input" id="find-subject" placeholder="something"> -->

            <button type="submit" class="btn btn-outline-success mt-2 mt-sm-0">Submit</button>
          </form>

          <p class="lead font-weight-bold">Case list</p>
          
          <table class='table table-striped bgc-transparent text-color'>
            <thead>
              <tr>
                <th>Student ID</th>
                <th>Region</th>
                <th>Subject</th>
                <th>Time</th>
                <th>Seniority</th>
                <th>Check</th>
                <th>Accept</th>
              </tr>
            </thead>
            <tbody>
              <?=$table3_text?>
            </tbody>
          </table>

        </div>

        <div class="row justify-content-center">
          <a href="afterlogin.php" type="button" class="btn btn-outline-light m-1">Go Back</a>
        </div>

      </div>
    </div>



  <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
  <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js" integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" crossorigin="anonymous"></script>

  <script>
    $(".chk").click(function(){
      $(this).next().submit();
    });
    $(".acp").click(function(){
      $(this).next().submit();
    });
  </script>
  </body>
</html>