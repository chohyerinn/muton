package com.example.myapplication

import android.os.Bundle
import android.widget.Toast
import com.example.myapplication.databinding.ActivityEditProfileBinding
import com.google.firebase.auth.FirebaseAuth

class EditProfileActivity : BaseActivity() {

    private lateinit var binding: ActivityEditProfileBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityEditProfileBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.backButton.setOnClickListener { finish() }
        loadProfile()
        binding.btnComplete.setOnClickListener {
            saveProfile()
        }
    }

    private fun loadProfile() {
        val user = FirebaseAuth.getInstance().currentUser
        val uid = user?.uid ?: return
        val fallbackName =
            user.displayName?.takeIf { it.isNotBlank() }
                ?: user.email?.substringBefore("@")
                ?: getString(R.string.settings_user_placeholder)
        FirebaseUserStore.loadProfile(this) { profile ->
            runOnUiThread {
                val displayName = profile.displayName.ifBlank { fallbackName }
                binding.txtUserName.text = displayName
                binding.inputDisplayName.setText("")
            }
        }
    }

    private fun saveProfile() {
        FirebaseAuth.getInstance().currentUser ?: return
        val displayName = binding.inputDisplayName.text.toString().trim()
        if (displayName.isBlank()) {
            Toast.makeText(this, R.string.settings_profile_empty, Toast.LENGTH_SHORT).show()
            return
        }

        FirebaseUserStore.saveProfile(
            context = this,
            displayName = displayName,
            onSuccess = {
                Toast.makeText(this, R.string.settings_profile_saved, Toast.LENGTH_SHORT).show()
                finish()
            },
            onFailure = { error ->
                Toast.makeText(
                    this,
                    getString(R.string.settings_profile_failed, error.localizedMessage ?: error.message.orEmpty()),
                    Toast.LENGTH_LONG,
                ).show()
            },
        )
    }
}
