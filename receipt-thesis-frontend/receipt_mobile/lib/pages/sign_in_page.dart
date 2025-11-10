import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class SignInPage extends StatefulWidget {
  const SignInPage({super.key});
  @override
  State<SignInPage> createState() => _SignInPageState();
}

class _SignInPageState extends State<SignInPage> {
  final emailC = TextEditingController();
  final passC = TextEditingController();
  final _formKey = GlobalKey<FormState>();
  bool loading = false;
  String? error;
  bool _isPasswordVisible = false;
  bool _rememberMe = false;

  @override
  void dispose() {
    emailC.dispose();
    passC.dispose();
    super.dispose();
  }

  Future<void> _signIn() async {
    if (!_validateForm()) return;
    setState(() {
      loading = true;
      error = null;
    });
    try {
      final resp = await Supabase.instance.client.auth.signInWithPassword(
        email: emailC.text.trim(),
        password: passC.text,
      );

      // Check if we got a session and token
      final session = resp.session;
      final user = resp.user;
      debugPrint("Supabase signIn response: user=${user?.id}");
      debugPrint("Access token: ${session?.accessToken}");

      if (session == null || session.accessToken.isEmpty) {
        throw Exception("Login failed: no session returned");
      }
    } catch (e) {
      setState(() => error = e.toString());
    } finally {
      if (mounted) setState(() => loading = false);
    }
  }

  Future<void> _signUp() async {
    if (!_validateForm()) return;
    setState(() {
      loading = true;
      error = null;
    });
    try {
      await Supabase.instance.client.auth.signUp(
        email: emailC.text.trim(),
        password: passC.text,
      );
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Check your email to confirm account.')),
      );
    } catch (e) {
      setState(() => error = e.toString());
    } finally {
      if (mounted) setState(() => loading = false);
    }
  }

  bool _validateForm() {
    final ok = _formKey.currentState?.validate() ?? false;
    if (!ok && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please fix the errors to continue.')),
      );
    }
    return ok;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: Center(
            child: Card(
              elevation: 8,
              margin: const EdgeInsets.all(24),
              child: Container(
                constraints: const BoxConstraints(maxWidth: 360),
                padding: const EdgeInsets.all(32),
                child: SingleChildScrollView(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const FlutterLogo(size: 96),
                      _gap(),
                      Text(
                        'Welcome to Receipt Manager',
                        style: Theme.of(context).textTheme.titleMedium,
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Enter your email and password to continue.',
                        style: Theme.of(context).textTheme.bodySmall,
                        textAlign: TextAlign.center,
                      ),
                      _gap(),
                      TextFormField(
                        controller: emailC,
                        enabled: !loading,
                        keyboardType: TextInputType.emailAddress,
                        decoration: const InputDecoration(
                          labelText: 'Email',
                          hintText: 'you@example.com',
                          prefixIcon: Icon(Icons.email_outlined),
                          border: OutlineInputBorder(),
                        ),
                        validator: (value) {
                          final text = value?.trim() ?? '';
                          if (text.isEmpty) {
                            return 'Please enter your email';
                          }
                          final emailRegex =
                              RegExp(r'^[\w\.\-+]+@[a-zA-Z0-9\-]+\.[a-zA-Z]+$');
                          if (!emailRegex.hasMatch(text)) {
                            return 'Enter a valid email';
                          }
                          return null;
                        },
                      ),
                      _gap(),
                      TextFormField(
                        controller: passC,
                        enabled: !loading,
                        obscureText: !_isPasswordVisible,
                        decoration: InputDecoration(
                          labelText: 'Password',
                          hintText: 'Enter your password',
                          border: const OutlineInputBorder(),
                          prefixIcon: const Icon(Icons.lock_outline),
                          suffixIcon: IconButton(
                            icon: Icon(_isPasswordVisible
                                ? Icons.visibility_off
                                : Icons.visibility),
                            onPressed: () => setState(
                              () => _isPasswordVisible = !_isPasswordVisible,
                            ),
                          ),
                        ),
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return 'Please enter your password';
                          }
                          if (value.length < 6) {
                            return 'Password must be at least 6 characters';
                          }
                          return null;
                        },
                      ),
                      _gap(),
                      CheckboxListTile(
                        value: _rememberMe,
                        onChanged: loading
                            ? null
                            : (val) {
                                if (val == null) return;
                                setState(() => _rememberMe = val);
                              },
                        dense: true,
                        controlAffinity: ListTileControlAffinity.leading,
                        contentPadding: EdgeInsets.zero,
                        title: const Text('Remember me'),
                      ),
                      if (error != null) ...[
                        const SizedBox(height: 8),
                        Text(error!, style: const TextStyle(color: Colors.red)),
                      ],
                      _gap(),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: loading ? null : _signIn,
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 12),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(6),
                            ),
                          ),
                          child: loading
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(strokeWidth: 2),
                                )
                              : const Text(
                                  'Sign in',
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                        ),
                      ),
                      const SizedBox(height: 12),
                      SizedBox(
                        width: double.infinity,
                        child: OutlinedButton(
                          onPressed: loading ? null : _signUp,
                          child: const Padding(
                            padding: EdgeInsets.symmetric(vertical: 10),
                            child: Text('Create account'),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _gap() => const SizedBox(height: 16);
}
