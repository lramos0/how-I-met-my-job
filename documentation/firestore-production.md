# Firestore production setup

The app now supports a Reddit-style community layer on top of Firebase Auth and Firestore. If Firebase is not configured, the same UI falls back to `localStorage` for development.

## Collections

- `users/{uid}`: public profile, headline, bio, and activity stats.
- `users/{uid}/private/jobState`: private saved/applied/hidden job state.
- `forumPosts/{postId}`: public forum threads with `companySlug`, title, body, author profile snapshot, score, and comment count.
- `forumComments/{commentId}`: public flat comments with `postId`, `companySlug`, optional `parentId`, score, and author profile snapshot.
- `forumVotes/{targetType_targetId_uid}`: private per-user vote records. The app reads only the signed-in user's votes.

## Required composite indexes

Create these indexes in the Firebase console if Firestore prompts for them:

- `forumPosts`: `companySlug` ascending, `status` ascending, `createdAt` descending.
- `forumComments`: `postId` ascending, `status` ascending.
- `forumComments`: `companySlug` ascending, `status` ascending, `createdAt` ascending. This is a fallback scan if post-scoped comment loading fails.
- `forumVotes`: `userUid` ascending, `companySlug` ascending.
- `forumPosts`: `authorUid` ascending, `status` ascending, `createdAt` descending.
- `forumComments`: `authorUid` ascending, `status` ascending, `createdAt` descending.

## Starter security rules

These rules are intended for a Firebase client-only MVP. They prevent anonymous writes, keep saved jobs private, make profiles/posts/comments public, and restrict vote documents to the signed-in user.

For stricter production hardening, move score/comment-count aggregation into Cloud Functions so clients only write `forumVotes` and `forumComments`; then remove direct client updates to `score` and `commentCount`.

```text
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    function signedIn() {
      return request.auth != null;
    }

    function owns(userId) {
      return signedIn() && request.auth.uid == userId;
    }

    function stringBetween(value, min, max) {
      return value is string && value.size() >= min && value.size() <= max;
    }

    function activePublic() {
      return resource.data.status == 'active';
    }

    match /users/{userId} {
      allow read: if true;
      allow create, update: if owns(userId)
        && request.resource.data.uid == userId;

      match /private/{docId} {
        allow read, write: if owns(userId);
      }
    }

    match /forumPosts/{postId} {
      allow read: if activePublic();

      allow create: if signedIn()
        && request.resource.data.authorUid == request.auth.uid
        && request.resource.data.status == 'active'
        && stringBetween(request.resource.data.companySlug, 1, 140)
        && stringBetween(request.resource.data.title, 1, 300)
        && stringBetween(request.resource.data.body, 0, 6000)
        && request.resource.data.score == 1
        && request.resource.data.commentCount == 0;

      allow update: if signedIn()
        && request.resource.data.diff(resource.data).affectedKeys().hasOnly(['score', 'commentCount', 'updatedAt'])
        && request.resource.data.score - resource.data.score >= -2
        && request.resource.data.score - resource.data.score <= 2
        && request.resource.data.commentCount - resource.data.commentCount >= 0
        && request.resource.data.commentCount - resource.data.commentCount <= 1;
    }

    match /forumComments/{commentId} {
      allow read: if activePublic();

      allow create: if signedIn()
        && request.resource.data.authorUid == request.auth.uid
        && request.resource.data.status == 'active'
        && stringBetween(request.resource.data.companySlug, 1, 140)
        && stringBetween(request.resource.data.postId, 1, 180)
        && stringBetween(request.resource.data.body, 1, 4000)
        && request.resource.data.score == 1;

      allow update: if signedIn()
        && request.resource.data.diff(resource.data).affectedKeys().hasOnly(['score', 'updatedAt'])
        && request.resource.data.score - resource.data.score >= -2
        && request.resource.data.score - resource.data.score <= 2;
    }

    match /forumVotes/{voteId} {
      allow get: if signedIn()
        && (!exists(/databases/$(database)/documents/forumVotes/$(voteId))
          || resource.data.userUid == request.auth.uid);

      allow list: if signedIn() && resource.data.userUid == request.auth.uid;

      allow create, update: if signedIn()
        && request.resource.data.userUid == request.auth.uid
        && request.resource.data.value in [-1, 1]
        && request.resource.data.targetType in ['post', 'comment']
        && stringBetween(request.resource.data.companySlug, 1, 140);

      allow delete: if signedIn() && resource.data.userUid == request.auth.uid;
    }
  }
}
```
