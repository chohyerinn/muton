package com.example.myapplication

import android.content.Context
import android.net.Uri
import com.google.firebase.FirebaseApp
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.UserProfileChangeRequest
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.storage.FirebaseStorage
import java.io.File

object FirebaseUserStore {
    data class UserProfile(
        val displayName: String,
    )

    fun isConfigured(context: Context): Boolean = FirebaseApp.getApps(context).isNotEmpty()

    fun toAuthEmail(identifier: String): String {
        val trimmed = identifier.trim()
        return if (trimmed.contains("@")) trimmed else "$trimmed@muton.local"
    }

    fun signUp(
        context: Context,
        identifier: String,
        password: String,
        voiceSampleFile: File? = null,
        onSuccess: () -> Unit,
        onFailure: (Exception) -> Unit,
    ) {
        if (!isConfigured(context)) {
            onFailure(IllegalStateException(context.getString(R.string.firebase_not_configured)))
            return
        }

        val displayName = identifier.trim()
        val authEmail = toAuthEmail(displayName)

        FirebaseAuth.getInstance()
            .createUserWithEmailAndPassword(authEmail, password)
            .addOnSuccessListener { result ->
                val uid = result.user?.uid.orEmpty()
                val createdAt = System.currentTimeMillis()

                val profileUpdates = UserProfileChangeRequest.Builder()
                    .setDisplayName(displayName)
                    .build()

                result.user?.updateProfile(profileUpdates)
                    ?.addOnCompleteListener {
                        uploadVoiceSampleIfNeeded(
                            uid = uid,
                            voiceSampleFile = voiceSampleFile,
                            onSuccess = { voiceSamplePath, voiceSampleUrl ->
                                val user = mutableMapOf<String, Any>(
                                    "uid" to uid,
                                    "displayName" to displayName,
                                    "authEmail" to authEmail,
                                    "createdAt" to createdAt,
                                )

                                if (!voiceSamplePath.isNullOrBlank()) {
                                    user["voiceSamplePath"] = voiceSamplePath
                                }
                                if (!voiceSampleUrl.isNullOrBlank()) {
                                    user["voiceSampleUrl"] = voiceSampleUrl
                                }

                                FirebaseFirestore.getInstance()
                                    .collection("users")
                                    .document(uid)
                                    .set(user)
                                    .addOnSuccessListener { onSuccess() }
                                    .addOnFailureListener(onFailure)
                            },
                            onFailure = onFailure,
                        )
                    }
                    ?.addOnFailureListener(onFailure)
            }
            .addOnFailureListener(onFailure)
    }

    private fun uploadVoiceSampleIfNeeded(
        uid: String,
        voiceSampleFile: File?,
        onSuccess: (String?, String?) -> Unit,
        onFailure: (Exception) -> Unit,
    ) {
        if (voiceSampleFile == null || !voiceSampleFile.exists() || voiceSampleFile.length() <= 0L) {
            onSuccess(null, null)
            return
        }

        val voicePath = "users/$uid/voice/signup_voice_note.m4a"
        val voiceRef = FirebaseStorage.getInstance().reference.child(voicePath)
        val voiceUri = Uri.fromFile(voiceSampleFile)

        voiceRef.putFile(voiceUri)
            .continueWithTask { uploadTask ->
                val exception = uploadTask.exception
                if (exception != null) {
                    throw exception
                }
                voiceRef.downloadUrl
            }
            .addOnSuccessListener { downloadUri ->
                onSuccess(voicePath, downloadUri.toString())
            }
            .addOnFailureListener { error ->
                onFailure(Exception(error))
            }
    }

    fun signIn(
        context: Context,
        identifier: String,
        password: String,
        onSuccess: () -> Unit,
        onFailure: (Exception) -> Unit,
    ) {
        if (!isConfigured(context)) {
            onFailure(IllegalStateException(context.getString(R.string.firebase_not_configured)))
            return
        }

        FirebaseAuth.getInstance()
            .signInWithEmailAndPassword(toAuthEmail(identifier), password)
            .addOnSuccessListener { onSuccess() }
            .addOnFailureListener(onFailure)
    }

    fun loadDisplayName(
        context: Context,
        onResult: (String) -> Unit,
    ) {
        loadProfile(context) { profile ->
            onResult(profile.displayName)
        }
    }

    fun loadProfile(
        context: Context,
        onResult: (UserProfile) -> Unit,
    ) {
        val auth = FirebaseAuth.getInstance()
        val user = auth.currentUser
        val fallbackName = user?.displayName
            ?.takeIf { it.isNotBlank() }
            ?: user?.email?.substringBefore("@")
            ?: context.getString(R.string.settings_user_placeholder)

        val uid = user?.uid
        if (uid.isNullOrBlank() || !isConfigured(context)) {
            onResult(UserProfile(fallbackName))
            return
        }

        FirebaseFirestore.getInstance()
            .collection("users")
            .document(uid)
            .get()
            .addOnSuccessListener { document ->
                onResult(
                    UserProfile(
                        displayName = document.getString("displayName").orEmpty().ifBlank { fallbackName },
                    ),
                )
            }
            .addOnFailureListener {
                onResult(UserProfile(fallbackName))
            }
    }

    fun saveProfile(
        context: Context,
        displayName: String,
        onSuccess: () -> Unit,
        onFailure: (Exception) -> Unit,
    ) {
        val user = FirebaseAuth.getInstance().currentUser
        val uid = user?.uid
        if (user == null || uid.isNullOrBlank() || !isConfigured(context)) {
            onFailure(IllegalStateException(context.getString(R.string.settings_password_user_missing)))
            return
        }

        val profileUpdates = UserProfileChangeRequest.Builder()
            .setDisplayName(displayName)
            .build()

        val payload = mapOf(
            "uid" to uid,
            "displayName" to displayName,
            "authEmail" to (user.email ?: ""),
        )

        FirebaseFirestore.getInstance()
            .collection("users")
            .document(uid)
            .set(payload)
            .addOnSuccessListener {
                user.updateProfile(profileUpdates)
                    .addOnSuccessListener { onSuccess() }
                    .addOnFailureListener(onFailure)
            }
            .addOnFailureListener(onFailure)
    }

    fun loadAuthEmail(
        context: Context,
        onResult: (String?) -> Unit,
    ) {
        val user = FirebaseAuth.getInstance().currentUser
        val fallbackEmail = user?.email
        val uid = user?.uid

        if (uid.isNullOrBlank() || !isConfigured(context)) {
            onResult(fallbackEmail)
            return
        }

        FirebaseFirestore.getInstance()
            .collection("users")
            .document(uid)
            .get()
            .addOnSuccessListener { document ->
                onResult(document.getString("authEmail").orEmpty().ifBlank { fallbackEmail })
            }
            .addOnFailureListener {
                onResult(fallbackEmail)
            }
    }
}
