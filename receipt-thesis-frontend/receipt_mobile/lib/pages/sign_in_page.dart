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
  bool loading = false;
  String? error;

  Future<void> _signIn() async {
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

      if (mounted) {
        Navigator.of(context).pushReplacementNamed('/home');
      }
    } catch (e) {
      setState(() => error = e.toString());
    } finally {
      if (mounted) setState(() => loading = false);
    }
  }

  Future<void> _signUp() async {
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Sign in')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(children: [
          TextField(
              controller: emailC,
              decoration: const InputDecoration(labelText: 'Email')),
          const SizedBox(height: 8),
          TextField(
              controller: passC,
              obscureText: true,
              decoration: const InputDecoration(labelText: 'Password')),
          const SizedBox(height: 12),
          if (error != null)
            Text(error!, style: const TextStyle(color: Colors.red)),
          const SizedBox(height: 12),
          Row(children: [
            Expanded(
                child: FilledButton(
                    onPressed: loading ? null : _signIn,
                    child: const Text('Sign in'))),
            const SizedBox(width: 12),
            Expanded(
                child: OutlinedButton(
                    onPressed: loading ? null : _signUp,
                    child: const Text('Sign up'))),
          ]),
        ]),
      ),
    );
  }
}
