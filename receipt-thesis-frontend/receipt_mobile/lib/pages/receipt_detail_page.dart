import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models.dart';
import '../api_client.dart';
import 'custom_labels_page.dart';

final apiProvider = Provider<ApiClient>((ref) => ApiClient());

class ReceiptDetailPage extends ConsumerStatefulWidget {
  final Receipt receipt;
  const ReceiptDetailPage({super.key, required this.receipt});

  @override
  ConsumerState<ReceiptDetailPage> createState() => _ReceiptDetailPageState();
}

class _ReceiptDetailPageState extends ConsumerState<ReceiptDetailPage> {
  late final TextEditingController storeCtl;
  late final TextEditingController dateCtl;
  late final TextEditingController totalCtl;
  String? category;

  static const builtinCats = [
    'Utilities',
    'Food',
    'Groceries',
    'Transportation',
    'Health & Wellness',
    'Others'
  ];

  List<CustomLabel> _customLabels = [];
  bool _loadingCategories = true;

  @override
  void initState() {
    super.initState();
    storeCtl = TextEditingController(text: widget.receipt.store ?? '');
    dateCtl = TextEditingController(text: widget.receipt.date ?? '');
    totalCtl = TextEditingController(
        text: widget.receipt.total?.toStringAsFixed(2) ?? '');
    category = widget.receipt.category ?? 'Others';
    _loadCustomLabels();
  }

  Future<void> _loadCustomLabels() async {
    try {
      final api = ref.read(apiProvider);
      final labels = await api.listCustomLabels();
      setState(() {
        _customLabels = labels;
        _loadingCategories = false;
      });
    } catch (e) {
      setState(() {
        _loadingCategories = false;
      });
    }
  }

  @override
  void dispose() {
    storeCtl.dispose();
    dateCtl.dispose();
    totalCtl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final api = ref.read(apiProvider);
    final t = double.tryParse(totalCtl.text);
    final ok = await api.updateReceipt(
      id: widget.receipt.id,
      store: storeCtl.text.isEmpty ? null : storeCtl.text,
      date: dateCtl.text.isEmpty ? null : dateCtl.text,
      total: t,
      category: category,
    );
    if (mounted) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text(ok ? 'Saved' : 'Failed')));
      if (ok) Navigator.pop(context, true); // Return true to indicate update
    }
  }

  Color _parseColor(String? hex) {
    if (hex == null || hex.isEmpty) return Colors.grey;
    try {
      final h = hex.replaceFirst('#', '');
      return Color(int.parse('FF$h', radix: 16));
    } catch (_) {
      return Colors.grey;
    }
  }

  List<DropdownMenuItem<String>> _buildCategoryItems() {
    final items = <DropdownMenuItem<String>>[];

    // Built-in categories section
    items.add(
      DropdownMenuItem<String>(
        enabled: false,
        value: '__builtin_header__',
        child: Text(
          'BUILT-IN',
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.bold,
            color: Colors.grey[600],
          ),
        ),
      ),
    );

    for (final cat in builtinCats) {
      items.add(
        DropdownMenuItem<String>(
          value: cat,
          child: Row(
            children: [
              Icon(
                _getCategoryIcon(cat),
                size: 20,
                color: Colors.grey[700],
              ),
              const SizedBox(width: 8),
              Text(cat),
            ],
          ),
        ),
      );
    }

    // Custom labels section (if any)
    if (_customLabels.isNotEmpty) {
      items.add(
        DropdownMenuItem<String>(
          enabled: false,
          value: '__custom_header__',
          child: Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(
              'CUSTOM',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                color: Colors.grey[600],
              ),
            ),
          ),
        ),
      );

      for (final label in _customLabels) {
        items.add(
          DropdownMenuItem<String>(
            value: label.name,
            child: Row(
              children: [
                Container(
                  width: 20,
                  height: 20,
                  decoration: BoxDecoration(
                    color: _parseColor(label.color),
                    shape: BoxShape.circle,
                  ),
                  child: Center(
                    child: Text(
                      label.name.isNotEmpty ? label.name[0].toUpperCase() : '?',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Text(label.name),
                const SizedBox(width: 4),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: Colors.blue[50],
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    'Custom',
                    style: TextStyle(fontSize: 10, color: Colors.blue[700]),
                  ),
                ),
              ],
            ),
          ),
        );
      }
    }

    return items;
  }

  IconData _getCategoryIcon(String category) {
    switch (category) {
      case 'Utilities':
        return Icons.electrical_services;
      case 'Food':
        return Icons.restaurant;
      case 'Groceries':
        return Icons.shopping_cart;
      case 'Transportation':
        return Icons.directions_car;
      case 'Health & Wellness':
        return Icons.medical_services;
      case 'Others':
        return Icons.category;
      default:
        return Icons.label;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Receipt Detail'),
        actions: [
          IconButton(
            icon: const Icon(Icons.label_outline),
            tooltip: 'Manage Custom Labels',
            onPressed: () async {
              await Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (ctx) => const CustomLabelsPage(),
                ),
              );
              // Reload custom labels when returning
              _loadCustomLabels();
            },
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(
          children: [
            TextField(
              controller: storeCtl,
              decoration: const InputDecoration(
                labelText: 'Store',
                prefixIcon: Icon(Icons.store),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: dateCtl,
              decoration: const InputDecoration(
                labelText: 'Date (YYYY-MM-DD)',
                prefixIcon: Icon(Icons.calendar_today),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: totalCtl,
              decoration: const InputDecoration(
                labelText: 'Total (₱)',
                prefixIcon: Padding(
                  padding: EdgeInsets.all(12),
                  child: Text('₱',
                      style:
                          TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                ),
              ),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 16),

            // Category section
            Row(
              children: [
                Text(
                  'Category',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                const Spacer(),
                TextButton.icon(
                  onPressed: () async {
                    await Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (ctx) => const CustomLabelsPage(),
                      ),
                    );
                    _loadCustomLabels();
                  },
                  icon: const Icon(Icons.add, size: 18),
                  label: const Text('Add Custom'),
                ),
              ],
            ),
            const SizedBox(height: 8),

            _loadingCategories
                ? const Center(child: CircularProgressIndicator())
                : DropdownButtonFormField<String>(
                    initialValue:
                        _isValidCategory(category) ? category : 'Others',
                    items: _buildCategoryItems(),
                    onChanged: (v) {
                      if (v != null && !v.startsWith('__') && v != category) {
                        setState(() => category = v);
                      }
                    },
                    decoration: InputDecoration(
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 8,
                      ),
                    ),
                    isExpanded: true,
                  ),

            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: _save,
              icon: const Icon(Icons.save),
              label: const Text('Save Changes'),
            ),

            const SizedBox(height: 16),

            // Info card
            Card(
              color: Colors.blue[50],
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: [
                    Icon(Icons.info_outline, color: Colors.blue[700]),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Need a category not listed? Create a custom label to track new receipt types.',
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.blue[900],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  bool _isValidCategory(String? cat) {
    if (cat == null) return false;
    if (builtinCats.contains(cat)) return true;
    return _customLabels.any((l) => l.name == cat);
  }
}
