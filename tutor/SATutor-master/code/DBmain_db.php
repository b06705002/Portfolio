<?php

echo "jizz";

?>
<?
session_start();

$region = $_POST['Region'];
$subject = $_POST['Subject'];
$time = $_POST['Time'];
$st = $_POST['ST'];
$uid = $_SESSION['username'];
echo $uid;

include("connect.php");



if($st == 'S')
{
	$sql = "SELECT * FROM student_table WHERE uid = '$uid'";
	$result = $mysqli->query($sql);
	$row = $result->fetch_row();
	$sid = $row[0]
	        //將帳號寫入session，方便驗證使用者身份
	//$con= new mysqli("localhost","root","XD221165","mydb");
	$sql = "INSERT into S_case_table (SID, Region, Subject, `Time`) values ('$sid', '$region','$subject', '$time')";

	$result = $mysqli->query($sql);
	if($result)
	{
	    echo '新增成功!';
	    echo '<meta http-equiv=REFRESH CONTENT=2;url=afterlogin.php>';
	}
	else
	{
	    echo '新增失敗!';
	    echo '<meta http-equiv=REFRESH CONTENT=2;url=afterlogin.php>';
	}
}
else if($st == 'T')
{
	$sql = "SELECT * FROM teacher_table WHERE uid = '$uid'";
	$result = $mysqli->query($sql);
	$row = $result->fetch_row();
	$tid = $row[0]
	        //將帳號寫入session，方便驗證使用者身份
	//$con= new mysqli("localhost","root","XD221165","mydb");
	$sql = "INSERT into T_case_table (TID, Region, Subject, `Time`) values ('$tid', '$region','$subject', '$time')";

	$result = $mysqli->query($sql);
	if($result)
	{
	    echo '新增成功!';
	    echo '<meta http-equiv=REFRESH CONTENT=2;url=afterlogin.php>';
	}
	else
	{
	    echo '新增失敗!';
	    echo '<meta http-equiv=REFRESH CONTENT=2;url=afterlogin.php>';
	}
}
else
{
	echo 'Wrong Choose';
	echo '<meta http-equiv=REFRESH CONTENT=1;url=afterlogin.php>';
}


?>