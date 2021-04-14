How to merge approved PRs
=========================

Requirements
--------------

- A remote called ``upstream`` pointing at the central IC repository.

- A remote configured for the author of the PR.

- After the PR is approved, make sure that it is ready to be merged (asking the authors if they didn't get in touch with you).


Steps
-------

#. Fetch ``upstream/master``.

#. Reset or rebase your master branch to ``upstream/master``.

#. Fetch the branch of the approved PR.

#. Create and check out a local branch on top of the PR branch.

#. Make sure the branch of the approved PR is rebased onto ``upstream/master``. If not:

   * Rebase the branch onto ``upstream/master``. If there are conflicts, ask the author to resolve them, unless they are obvious.

   * Push to the branch of the PR and wait until the tests finish.

#. Checkout your local ``master``.

#. Merge the PR branch into your local ``master``, making sure that the merge commit conforms to our requirements. Here are the steps needed to make the merge happen:

   * Disallow fast forward merging: we want an explicit merge commit for each PR.

   * Edit the commit message so that it has the following format:

   <PR number>  <PR title>

   <PR url>

   [author: <author's id>]

   <PR description>  (This is usually the whole first comment in the PR)

   [reviewer: <approver's id>]

   <Reviewer comment> (comment the reviewer left on GitHub)


   * Set the reviewer as the author of merge commit.

#. Push the merge commit to ``upstream/master``.

#. Delete the local branch you created in step 4.


Detailed mechanics
--------------------

#. **CLI:** ``git fetch upstream master``, **Magit:** ``f o upstream RET master RET``

#. Reset or rebase your master branch to ``upstream/master``.

#. Fetch the branch of the approved PR.

#. Create and check out a local branch on top of the PR branch.

#. Make sure the branch of the approved PR is rebased onto ``upstream/master``. If not:

   * Rebase the branch onto ``upstream/master``. If there are conflicts, ask the author to resolve them, unless they are obvious.

   * Push to the branch of the PR and wait until the tests finish.

#. Checkout your local ``master``.

#. Merge the PR branch into your local ``master``, making sure that the merge commit conforms to our requirements. Here are the steps needed to make the merge happen:

   * Disallow fast forward merging: we want an explicit merge commit for each PR.

   * Edit the commit message so that it has the following format:

   <PR number>  <PR title>

   <PR url>

   [author: <author's id>]

   <PR description>  (This is usually the whole first comment in the PR)

   [reviewer: <approver's id>]

   <Reviewer comment> (comment the reviewer left on GitHub)


   * Set the reviewer as the author of merge commit.

#. Push the merge commit to ``upstream/master``.

#. Delete the local branch you created in step 4.

