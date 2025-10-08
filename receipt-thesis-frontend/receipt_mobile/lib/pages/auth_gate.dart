// lib/auth_gate.dart
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'home_page.dart';
import 'sign_in_page.dart';

class AuthGate extends StatelessWidget {
  const AuthGate({super.key});

  @override
  Widget build(BuildContext context) {
    final session = Supabase.instance.client.auth.currentSession;

    return StreamBuilder<AuthState>(
      stream: Supabase.instance.client.auth.onAuthStateChange,
      builder: (context, snapshot) {
        // If we already have a session (app just started and user logged in), go Home
        if (session != null) {
          return const HomePage();
        }
        // Listen to changes
        if (snapshot.hasData) {
          final data = snapshot.data!;
          if (data.event == AuthChangeEvent.signedIn) {
            return const HomePage();
          }
          if (data.event == AuthChangeEvent.signedOut) {
            return const SignInPage();
          }
        }
        // Default (no session): Sign-in
        return const SignInPage();
      },
    );
  }
}
