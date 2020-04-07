# Coding conventions

Try to adhere to these coding conventions if possible. They ensure a cleanly
readable code throughout the project:

*   Do not use `auto`
*   Use two spaces for indentation
*   Use `UpperCamelCase` for class names
*   Use `UpperCamelCase` for methods
*   Use `snake_case` for variables
*   Add a `m_` prefix to member variables
*   Please handle your errors. Do not use `assert`
    * A common way of handling errors is for a tools' `Initialise()`, `Execute()`, and `Finalise()` methods to return `false`. This will inform the framework of the failure allowing it to decide on the action to take. It can stop immediately, try to recover, or close the program graciously, running finalise on each tool rather than a hard stop. This lets you disconnect hardware connections etc. When returning false, send a log message with verbosity level 0, to ensure that it is always logged.
*   Explain your code with helpful (!) comments, ideally using Doxygen syntax
*   Classes, functions, methods, and if statements should place the opening brace at the end of the line, rather than on a new line
    * e.g. `if(1) {`
    * not
```
if
{
```
*   Short if statements should be put on one line
    * e.g. `if(1) return true;`
    * not
```
if(1)
return true
```
*   Functions that are usable by multiple tools, but don't live in the DataModel itself should go in `DataModel/Utilities.{cpp,h}` and live inside the `util` namespace
    * Classes should also go into that folder and namespace, but should have their own files e.g. `Stopwatch.h` & `Stopwatch.cpp`
