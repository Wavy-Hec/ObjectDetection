# Planning Phase Command

**Purpose:** Break down design into executable tasks  
**Input:** design.md and PRD.md  
**Output:** plan.md with task breakdown  
**When to use:** After design is approved, before implementation

---

## Prompt for AI Assistant

```
You are a project planner for the Real-Time Object Tracking project.

CONTEXT:
- Review PAD.md for overall architecture and standards
- Review feature-[X]/PRD.md for requirements
- Review feature-[X]/design.md for technical design
- Understand task dependencies and critical path

TASK:
Create a detailed project plan (plan.md) that breaks the feature into tasks:

1. **Task List**: 
   - Each task should be 1-4 hours of focused work
   - Tasks should have clear inputs and outputs
   - Dependencies should be explicit
   
2. **For Each Task**:
   - **Task Name**: Clear, action-oriented name
   - **Description**: What needs to be accomplished
   - **Dependencies**: What must be completed first
   - **Steps**: Detailed implementation steps
   - **Acceptance Criteria**: Definition of done
   - **Estimated Time**: Realistic time estimate
   
3. **Task Organization**:
   - Group by module or component
   - Order by dependencies
   - Identify critical path
   - Note which can be done in parallel
   
4. **Overall Timeline**:
   - Phases with milestones
   - Total estimated duration
   - Key deliverables per phase

REQUIREMENTS:
- Tasks must align with design.md
- Each task must be independently testable
- Dependencies must form a valid DAG (no cycles)
- Time estimates must be realistic
- Include testing and documentation tasks

OUTPUT FORMAT:
- One section per task
- Clear numbering (Task 1, Task 2, etc.)
- Dependency graph or list
- Timeline with milestones

Please create the project plan now.
```

---

## AI Assistant Instructions

1. **Read design.md** to understand what needs to be built
2. **Identify modules** that need to be created or modified
3. **Break into small tasks** - aim for 1-4 hour chunks
4. **Map dependencies** - create a dependency graph
5. **Estimate time** - be realistic, add buffer
6. **Add validation** - include testing in each task
7. **Create plan.md** with all tasks documented

---

## Task Breakdown Guidelines

### Good Task Size
- ✅ **1-4 hours**: "Implement Detection class with bbox, label, confidence"
- ✅ **2-3 hours**: "Write unit tests for detector.py with 80% coverage"
- ❌ **Too large**: "Build the entire tracking system" (break it down!)
- ❌ **Too small**: "Import numpy" (combine with other setup)

### Task Dependencies
```
Task 1: Setup (no dependencies)
    ↓
Task 2: Create detector.py (depends on Task 1)
    ↓
Task 3: Create tracker.py (depends on Task 1)
    ↓
Task 4: Integrate in main.py (depends on Tasks 2, 3)
    ↓
Task 5: Testing (depends on Task 4)
```

### Task Template
```markdown
## Task [N]: [Action-Oriented Name]

**Description:** [What needs to be accomplished]

**Dependencies:** 
- Task [X]: [Name]
- [Other dependencies]

**Steps:**
1. [Specific step]
2. [Specific step]
3. [Specific step]

**Acceptance Criteria:**
- [ ] [Measurable outcome]
- [ ] [Measurable outcome]

**Estimated Time:** [X hours]
```

---

## Checklist

Before starting implementation:

- [ ] All tasks are clearly defined
- [ ] Dependencies are mapped correctly
- [ ] No circular dependencies
- [ ] Time estimates are realistic
- [ ] Each task has acceptance criteria
- [ ] Testing tasks are included
- [ ] Documentation tasks are included
- [ ] Plan is reviewed and approved

---

## Common Mistakes to Avoid

1. **Tasks too large**: Breaking into only 2-3 huge tasks
   - ✅ Better: 5-10 smaller, focused tasks

2. **Missing dependencies**: Not documenting what depends on what
   - ✅ Better: Explicit dependency list for each task

3. **Vague acceptance criteria**: "Code works"
   - ✅ Better: "Unit tests pass, FPS > 30, MOTA > 0.80"

4. **No testing tasks**: Only implementation, no validation
   - ✅ Better: Include unit tests, integration tests, evaluation

5. **Unrealistic estimates**: "Build entire system in 2 hours"
   - ✅ Better: Realistic estimates with buffer for debugging

---

## Example Usage

```bash
# Step 1: Ensure you have design
cat feature-2/design.md
cat feature-2/PRD.md

# Step 2: Use the prompt with AI assistant
# [Copy prompt from above]

# Step 3: Save output to plan.md
# [AI generates plan.md]

# Step 4: Review task breakdown
# Verify dependencies make sense
# Check time estimates are reasonable
# Ensure all requirements covered

# Step 5: Create individual task files
mkdir -p feature-2/tasks
# Generate task-1.md, task-2.md, etc.
```

---

## Integration with Workflow

```
design.md (Architecture)
    ↓
[USE THIS COMMAND]
    ↓
plan.md (Task Breakdown)
    ↓
tasks/*.md (Individual Task Files)
    ↓
Implementation
```

---

**Last Updated:** 2025-01-27  
**Version:** 1.0
