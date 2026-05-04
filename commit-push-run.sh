

# I would like you to commit the changes in the current git repository with a commit message provided as an argument to the script, push the changes to the remote repository, and then run a specified command. If no commit message is provided, use "Auto-commit" as the default message. If no command is provided, do not run any command after pushing.

#!/bin/bash
# Usage: ./commit-push-run.bash "Your commit message"

COMMIT_MESSAGE=${1:-"Auto-commit"}

# Commit changes
git add -a
git commit -m "$COMMIT_MESSAGE"
git push

# Source - https://superuser.com/a
# Posted by Igor Feghali, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-20, License - CC BY-SA 3.0

ssh -A erikmam@narval.alliancecan.ca << 'ENDSSH'
  cd /home/erikmam/projects/def-scoulomb/erikmam/Locally_consistent_SCAFFOLD-GS && git pull && sbatch --account=def-scoulomb ./launch.slurm
ENDSSH
