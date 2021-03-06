<?php

include("connect.php");

$uid = $_POST['UID'];
$username = $_POST['Username'];
$sex = $_POST['Sex'];
$age = (int)$_POST['Age'];
$mail = $_POST['Mail'];
$passwd = $_POST['Password'];



if($uid != null && $passwd != null)
{
	$sql = "INSERT into member_table (uid, username, sex, age, mail, password) values ('$uid', '$username', '$sex', '$age', '$mail', '$passwd')";
    $result = $mysqli->query($sql);
//$con= new mysqli("localhost","root","XD221165","mydb");
	if($result)
    {
        //echo '新增成功!';
        echo '<meta http-equiv=REFRESH CONTENT=2;url=index.php>';
    }
    else
    {
        die($mysqli -> error);
        echo '新增失敗!';
        echo '<meta http-equiv=REFRESH CONTENT=2;url=index.php>';
    }
}
else
{
	echo '還想駭我阿';
	echo '<meta http-equiv=REFRESH CONTENT=2;url=index.php>';
}