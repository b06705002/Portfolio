<?php

$db_server = "localhost";

$db_name = "mydb";

$db_user = "root";

$db_passwd = "1230";

$mysqli = new mysqli("localhost", "root", "1230", "mydb");


if ($mysqli -> connect_error) {
    die("連線失敗: " . $mysqli->connect_error);
}
//echo "連線成功";
$mysqli->query("SET NAMES 'utf8'");
/*
if(!@mysqli_connect($db_server, $db_user, $db_passwd))
	die("無法對資料庫連線");

mysqli_query("SET NAMES utf8");

if(!@mysqli_select_db($db_name))
        die("無法使用資料庫");
*/


?> 
