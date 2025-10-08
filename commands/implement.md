# Implementation Phase Command

**Purpose:** Implement a specific task from the plan  
**Input:** Task specification from tasks/task-[N].md  
**Output:** Working code with tests  
**When to use:** During development, one task at a time

---

## Prompt for AI Assistant

```
You are a Python developer working on the Real-Time Object Tracking project.

CONTEXT:
- Review PAD.md for coding standards and architecture
- Review feature-[X]/design.md for technical design
- Review feature-[X]/tasks/task-[N].md for this specific task
- Review existing code to understand patterns and conventions

TASK:
Implement the functionality specified in task-[N].md:

1. **Read Requirements**:
   - Understand what the task accomplishes
   - Identify inputs and outputs
   - Review acceptance criteria
   
2. **Plan Implementation**:
   - Identify files to create/modify
   - Design data structures
   - Plan function signatures
   
3. **Write Code**:
   - Follow PEP 8 style guidelines
   - Add docstrings to all public functions/classes
   - Include type hints where appropriate
   - Add inline comments for complex logic
   - Handle errors gracefully
   
4. **Write Tests**:
   - Unit tests for all public functions
   - Test edge cases and error conditions
   - Aim for >80% code coverage
   
5. **Validate**:
   - Run the code and verify it works
   - Run tests and ensure they pass
   - Check code style with linter
   - Verify acceptance criteria are met

REQUIREMENTS:
- Code must align with PAD.md architecture
- Follow existing code patterns and style
- All functions must have docstrings
- All tests must pass
- Acceptance criteria must be met

OUTPUT:
- Complete, working code files
- Test files with passing tests
- Brief summary of what was implemented

Please implement task-[N] now.
```

---

## AI Assistant Instructions

1. **Understand the task** - read task file completely
2. **Review context** - look at PAD, design, existing code
3. **Plan before coding** - think about structure
4. **Write code incrementally** - one function at a time
5. **Test as you go** - don't wait until the end
6. **Validate thoroughly** - run all tests
7. **Document changes** - update relevant docs

---

## Implementation Workflow

### Step 1: Read and Understand
```bash
# Read the task specification
cat feature-2/tasks/task-1.md

# Review related design
cat feature-2/design.md

# Check existing code patterns
cat src/*.py
```

### Step 2: Create/Modify Files
```python
# Follow this structure:

"""
Module description.

This module provides [functionality].
"""

import [libraries]
from typing import [types]


class ClassName:
    """Class description.
    
    Attributes:
        attr1: Description
        attr2: Description
    """
    
    def __init__(self, param1: Type1, param2: Type2):
        """Initialize the class.
        
        Args:
            param1: Description
            param2: Description
        """
        self.attr1 = param1
        self.attr2 = param2
    
    def method_name(self, param: Type) -> ReturnType:
        """Method description.
        
        Args:
            param: Description
        
        Returns:
            Description of return value
        
        Raises:
            ExceptionType: When this happens
        """
        # Implementation
        pass
```

### Step 3: Write Tests
```python
# tests/test_module.py

import pytest
from src.module import ClassName


def test_basic_functionality():
    """Test basic functionality."""
    obj = ClassName(param1, param2)
    result = obj.method_name(input)
    assert result == expected


def test_edge_case():
    """Test edge case."""
    # Test boundary conditions


def test_error_handling():
    """Test error handling."""
    with pytest.raises(ExceptionType):
        # Code that should raise exception
```

### Step 4: Validate
```bash
# Run the code
python src/module.py

# Run tests
pytest tests/test_module.py -v

# Check coverage
pytest tests/ --cov=src --cov-report=term

# Check style (optional)
flake8 src/module.py
```

---

## Code Quality Checklist

Before marking task complete:

- [ ] All required functions/classes implemented
- [ ] Code follows PEP 8 style guidelines
- [ ] All public APIs have docstrings
- [ ] Type hints added where appropriate
- [ ] Error handling is robust
- [ ] Edge cases are handled
- [ ] Unit tests written and passing
- [ ] Code coverage >80%
- [ ] Acceptance criteria from task file met
- [ ] No obvious bugs or issues
- [ ] Code reviewed (self or peer)

---

## Common Mistakes to Avoid

1. **No docstrings**: Missing documentation
   - ✅ Better: Every function/class has clear docstring

2. **No error handling**: Code crashes on bad input
   - ✅ Better: Validate inputs, raise meaningful exceptions

3. **No tests**: "It works when I run it"
   - ✅ Better: Comprehensive unit tests

4. **Hardcoded values**: Magic numbers throughout code
   - ✅ Better: Named constants or configuration

5. **Skipping validation**: Not running tests before committing
   - ✅ Better: Always validate before moving on

---

## Debugging Tips

### Issue: Code doesn't work as expected
```bash
# Add debug prints
print(f"DEBUG: variable = {variable}")

# Use Python debugger
import pdb; pdb.set_trace()

# Check types
print(f"Type: {type(variable)}")
```

### Issue: Tests failing
```bash
# Run single test with verbose output
pytest tests/test_module.py::test_name -v -s

# Show print statements
pytest tests/test_module.py -v -s

# Check test in isolation
pytest tests/test_module.py::test_name --pdb
```

### Issue: Import errors
```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Install missing package
pip install package_name

# Verify installation
pip list | grep package
```

---

## Example Usage

```bash
# Step 1: Read task
cat feature-2/tasks/task-3.md

# Step 2: Implement with AI assistant
# [Use prompt above]

# Step 3: Validate implementation
python src/tracker.py  # Run standalone
pytest tests/test_tracker.py -v  # Run tests
pytest tests/ --cov=src --cov-report=term  # Check coverage

# Step 4: Commit when complete
git add src/tracker.py tests/test_tracker.py
git commit -m "Implement SORT tracker (Task 3)"
```

---

## Integration with Workflow

```
tasks/task-[N].md (Specification)
    ↓
[USE THIS COMMAND]
    ↓
src/*.py (Implementation)
tests/*.py (Tests)
    ↓
Validation (tests pass)
    ↓
Next Task
```

---

**Last Updated:** 2025-01-27  
**Version:** 1.0
