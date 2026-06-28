package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.text.method.HideReturnsTransformationMethod
import android.text.method.PasswordTransformationMethod
import android.widget.Toast
import com.example.myapplication.databinding.ActivityLoginBinding

class LoginActivity : BaseActivity() {

    private lateinit var binding: ActivityLoginBinding
    private var isPasswordVisible = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLoginBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.inputPassword.transformationMethod = PasswordTransformationMethod.getInstance()

        binding.btnPasswordVisibility.setOnClickListener {
            isPasswordVisible = !isPasswordVisible
            binding.inputPassword.transformationMethod =
                if (isPasswordVisible) {
                    HideReturnsTransformationMethod.getInstance()
                } else {
                    PasswordTransformationMethod.getInstance()
                }
            binding.btnPasswordVisibility.alpha = if (isPasswordVisible) 1f else 0.55f
            binding.inputPassword.setSelection(binding.inputPassword.text?.length ?: 0)
        }

        binding.btnPasswordVisibility.alpha = 0.55f

        binding.btnLogin.setOnClickListener {
            val identifier = binding.inputEmail.text.toString().trim()
            val password = binding.inputPassword.text.toString()

            if (identifier.isBlank() || password.isBlank()) {
                Toast.makeText(this, R.string.login_empty_fields, Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            FirebaseUserStore.signIn(
                context = this,
                identifier = identifier,
                password = password,
                onSuccess = {
                    ConversationRecordStore.syncFromFirebase(this)
                    runOnUiThread {
                        startActivity(Intent(this, HomeActivity::class.java))
                        finish()
                    }
                },
                onFailure = { error ->
                    Toast.makeText(
                        this,
                        getString(R.string.login_failed, error.localizedMessage ?: error.message.orEmpty()),
                        Toast.LENGTH_LONG,
                    ).show()
                },
            )
        }

        binding.txtGoSignUp.setOnClickListener {
            startActivity(Intent(this, SignUpActivity::class.java))
        }
    }
}
