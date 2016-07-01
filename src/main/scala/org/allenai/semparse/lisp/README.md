This package contains code that makes working with jklol's lisp interpreter easier.  We define an
`Environment` class here, which is very similar to jklol's `Environment`, but this lets us keep it
in memory and interpret lisp statements programmatically, which is a much nicer interface than
having to work through a shell pipe.  This `Environment` also lets us extend the built-in
functions in jklol much easier, without having to modify jklol to get new lisp commands.
