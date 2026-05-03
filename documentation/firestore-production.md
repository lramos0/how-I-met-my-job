# Firestore production setup

The app now supports a Reddit-style community layer on top of Firebase Auth and Firestore. If Firebase is not configured, the same UI falls back to `localStorage` for development.

## Create the database first

Create a **Firestore Database** for the Firebase project before testing Google sign-in. This is not a SQL table and not Realtime Database. In the Firebase console, choose Firestore Database, create database, use Native mode if prompted, start in production mode, and choose your region.

You do not need to manually create the collections below. The first successful profile save, saved job, thread, comment, or vote will create the documents automatically.

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

These same indexes are also included in the repo root as `firestore.indexes.json` for Firebase CLI deploys.

## Option A: Create indexes in the Firebase console

1. Open Firebase Console.
2. Go to **Build > Firestore Database**.
3. Open the **Indexes** tab.
4. In **Composite indexes**, click **Add index**.
5. Set the collection ID and fields exactly as listed above.
   Type field paths manually without spaces. For example, use `status`, not `status `.
6. Leave query scope as **Collection**.
7. Click **Create**.
8. Wait until each index shows as enabled. Index builds can take a few minutes.

## Option B: Deploy indexes with the Firebase CLI

If you have the Firebase CLI connected to this project, deploy the checked-in index file:

```bash
firebase deploy --only firestore:indexes
```

If the CLI has not been initialized for this repo yet, run:

```bash
firebase init firestore
```

When prompted for the indexes file, use:

```text
firestore.indexes.json
```

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
