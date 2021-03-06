<?php

session_start();
include("connect.php");

$uid = $_POST['UID'];
$seniority = $_POST['Seniority'];
$edu = $_POST['Education'];
$pw = $_POST['Password'];

$sql = "SELECT * FROM member_table WHERE uid = '$uid'";

$result = $mysqli->query($sql);
//$con= new mysqli("localhost","root","XD221165","mydb");
$row = $result->fetch_row();




if($uid != null && $pw != null && $row[0] == $uid && $row[5] == $pw)
{
        //將帳號寫入session，方便驗證使用者身份
    $_SESSION['teachername'] = $uid;
    $sql = "INSERT into teacher_table (Seniority, Education, UID) values ('$seniority', '$edu','$uid')";
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
    echo '登入失敗!';
    echo '<meta http-equiv=REFRESH CONTENT=1;url=index.php>';
}
?>
