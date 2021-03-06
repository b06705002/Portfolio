<?php

session_start();

$uid = $_POST['UID'];
$pw = $_POST['password'];
include("connect.php");
$sql = "SELECT * FROM member_table WHERE uid = '$uid'";

$result = $mysqli->query($sql);
//$con= new mysqli("localhost","root","XD221165","mydb");
$row = $result->fetch_row();

if(isset($_SESSION['username']))
{
	echo '重複登入';
	echo '<meta http-equiv=REFRESH CONTENT=1;url=afterlogin.php>';
}


else if($uid != null && $pw != null && $row[0] == $uid && $row[5] == $pw)
{
        //將帳號寫入session，方便驗證使用者身份
    $_SESSION['username'] = $uid;
    $_SESSION['teachername'] = $uid;
    $_SESSION['studentname'] = $uid;
    //echo '登入成功!';
    echo '<meta http-equiv=REFRESH CONTENT=1;url=afterlogin.php>';
}

else
{
    echo '登入失敗!';
    echo '<meta http-equiv=REFRESH CONTENT=1;url=index.php>';
}
?>