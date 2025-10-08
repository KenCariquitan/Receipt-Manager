import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'pages/sign_in_page.dart';
import 'pages/home_page.dart'; // your existing main screen

class AuthGate extends StatefulWidget {
  const AuthGate({super.key});
  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  Session? _session;

  @override
  void initState() {
    super.initState();
    final auth = Supabase.instance.client.auth;
    _session = auth.currentSession;
    auth.onAuthStateChange.listen((data) {
      setState(() => _session = data.session);
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_session == null) return const SignInPage();
    return const HomePage(); // your app with tabs/pages
  }
}
