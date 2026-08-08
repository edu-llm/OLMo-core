$c setvar |- P $.
$v x y z $.
vx $f setvar x $.
vy $f setvar y $.
vz $f setvar z $.
${
  $d x y $.
  pair $a |- P x y $.
$}
bad $p |- P z z $= ( pair ) AAB $.
