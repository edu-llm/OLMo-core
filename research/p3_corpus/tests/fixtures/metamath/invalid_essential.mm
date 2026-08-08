$c wff |- $.
$v ph ps $.
wph $f wff ph $.
wps $f wff ps $.
${
  rule.1 $e |- ph $.
  rule $a |- ps $.
$}
${
  bad.1 $e |- ps $.
  bad $p |- ps $= ( wph rule ) CABD $.
$}
