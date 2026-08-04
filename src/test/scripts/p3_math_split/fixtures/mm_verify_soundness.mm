$c wff class setvar |- ( ) -> <-> P $.
$v ph ps ch th a b c X Z A x y z w $.
wph $f wff ph $.
wps $f wff ps $.
wch $f wff ch $.
wth $f wff th $.
wa $f wff a $.
wb $f wff b $.
wc $f wff c $.
wX $f wff X $.
wZ $f wff Z $.
cA $f class A $.
vx $f setvar x $.
vy $f setvar y $.
vz $f setvar z $.
vw $f setvar w $.

wi $a wff ( ph -> ps ) $.
ax-1 $a |- ( ph -> ( ps -> ph ) ) $.
${
  ax-mp.1 $e |- ph $.
  ax-mp.2 $e |- ( ph -> ps ) $.
  ax-mp $a |- ps $.
$}
${
  syl.1 $e |- ( ph -> ps ) $.
  syl.2 $e |- ( ps -> ch ) $.
  syl $a |- ( ph -> ch ) $.
$}
${
  from-one.1 $e |- ph $.
  from-one $a |- ph $.
$}
${
  dup.1 $e |- ph $.
  dup.2 $e |- ph $.
  dup $a |- ph $.
$}
emit $a |- ph $.
class-emit $a |- A $.

${
  $d x y $.
  pair $a |- P x y $.
$}
self-pair $a |- P x x $.

target-rule $a |- ( a -> ( c -> ( b -> a ) ) ) $.
ac-rule $a |- ( a -> c ) $.
bicond-rule $a |- ( a <-> b ) $.
xz-rule $a |- ( X -> Z ) $.
target $p |- ( a -> ( c -> ( b -> a ) ) ) $=
  wa wb wc target-rule
$.
ac-target $p |- ( a -> c ) $= wa wc ac-rule $.
bicond-target $p |- ( a <-> b ) $= wa wb bicond-rule $.
xz-target $p |- ( X -> Z ) $= wX wZ xz-rule $.
b-target $p |- b $= wb emit $.
dup-target $p |- ( a -> ( b -> a ) ) $= wa wb ax-1 $.

${
  th.1 $e |- a $.
  th.2 $e |- ( a -> b ) $.
  mp-target $p |- b $= wa wb th.1 th.2 ax-mp $.
$}

bad-rebind-rule $a |- ( ps -> Z ) $.
${
  bad-rebind.1 $e |- ( X -> X ) $.
  bad-rebind.2 $e |- ( X -> Z ) $.
  bad-rebind $p |- ( ps -> Z ) $= wps wZ bad-rebind-rule $.
$}

${
  good-rebind.1 $e |- ( ps -> X ) $.
  good-rebind.2 $e |- ( X -> Z ) $.
  good-rebind $p |- ( ps -> Z ) $=
    wps wX wZ good-rebind.1 good-rebind.2 syl
  $.
$}

class-target $p |- A $= cA class-emit $.
bad-d-target $p |- P z z $= vz self-pair $.
${
  $d z w $.
  good-d-target $p |- P z w $= vz vw pair $.
$}

${
  local-target.1 $e |- a $.
  local-target $p |- a $= local-target.1 $.
$}
