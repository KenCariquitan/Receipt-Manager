import 'package:flutter/material.dart';
import '../models.dart';
import '../api_client.dart';
import '../pages/custom_labels_page.dart';

class ReceiptCard extends StatefulWidget {
  final UploadResult res;
  final String? selectedCategory;
  final ValueChanged<String?>? onCategoryChanged;
  final TextEditingController storeController;
  final TextEditingController totalController;
  final TextEditingController dateController;

  const ReceiptCard({
    super.key,
    required this.res,
    this.selectedCategory,
    this.onCategoryChanged,
    required this.storeController,
    required this.totalController,
    required this.dateController,
  });

  @override
  State<ReceiptCard> createState() => _ReceiptCardState();
}

class _ReceiptCardState extends State<ReceiptCard> {
  static const builtinCats = [
    'Utilities',
    'Food',
    'Groceries',
    'Transportation',
    'Health & Wellness',
    'Others'
  ];

  List<CustomLabel> _customLabels = [];
  bool _loadingLabels = true;

  @override
  void initState() {
    super.initState();
    _loadCustomLabels();
  }

  Future<void> _loadCustomLabels() async {
    try {
      final api = ApiClient();
      final labels = await api.listCustomLabels();
      if (mounted) {
        setState(() {
          _customLabels = labels;
          _loadingLabels = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _loadingLabels = false;
        });
      }
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

  List<DropdownMenuItem<String>> _buildCategoryItems() {
    final items = <DropdownMenuItem<String>>[];

    // Built-in categories
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
      // Add a visual separator
      items.add(
        DropdownMenuItem<String>(
          enabled: false,
          value: '__divider__',
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Row(
              children: [
                Expanded(child: Divider(color: Colors.grey[400])),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  child: Text(
                    'Custom',
                    style: TextStyle(
                      fontSize: 11,
                      color: Colors.grey[600],
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
                Expanded(child: Divider(color: Colors.grey[400])),
              ],
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
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Text(label.name),
              ],
            ),
          ),
        );
      }
    }

    return items;
  }

  bool _isValidCategory(String? cat) {
    if (cat == null) return false;
    if (builtinCats.contains(cat)) return true;
    return _customLabels.any((l) => l.name == cat);
  }

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final currentCategory =
        widget.selectedCategory ?? widget.res.category ?? 'Others';

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ValueListenableBuilder<TextEditingValue>(
              valueListenable: widget.storeController,
              builder: (context, value, _) {
                final storeName = value.text.isNotEmpty
                    ? value.text
                    : (widget.res.store ?? 'Unknown Store');
                return Text(storeName, style: textTheme.titleLarge);
              },
            ),
            const SizedBox(height: 12),
            TextField(
              controller: widget.storeController,
              decoration: const InputDecoration(
                labelText: 'Store / Merchant',
                helperText: 'Edit if the merchant name is incorrect.',
                prefixIcon: Icon(Icons.store),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: widget.totalController,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              decoration: InputDecoration(
                labelText: 'Total Amount (₱)',
                helperText: widget.res.total != null
                    ? 'Detected total: ₱${widget.res.total!.toStringAsFixed(2)}'
                    : 'Enter the total from the receipt.',
                prefixIcon: const Padding(
                  padding: EdgeInsets.all(12),
                  child: Text('₱',
                      style:
                          TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                ),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: widget.dateController,
              keyboardType: TextInputType.datetime,
              decoration: InputDecoration(
                labelText: 'Receipt Date (YYYY-MM-DD)',
                helperText: widget.res.date != null
                    ? 'Detected date: ${widget.res.date}'
                    : 'Enter an ISO date if missing.',
                prefixIcon: const Icon(Icons.calendar_today),
              ),
            ),
            const SizedBox(height: 16),

            // Category section
            Row(
              children: [
                Text('Category', style: textTheme.titleSmall),
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
                  icon: const Icon(Icons.add, size: 16),
                  label:
                      const Text('Add Custom', style: TextStyle(fontSize: 12)),
                ),
              ],
            ),
            Text(
              'Predicted: ${widget.res.category ?? 'No model'}',
              style: textTheme.bodySmall?.copyWith(color: Colors.grey[600]),
            ),
            const SizedBox(height: 8),

            _loadingLabels
                ? const Center(child: CircularProgressIndicator())
                : DropdownButtonFormField<String>(
                    initialValue: _isValidCategory(currentCategory)
                        ? currentCategory
                        : 'Others',
                    items: _buildCategoryItems(),
                    onChanged: (v) {
                      if (v != null && !v.startsWith('__')) {
                        widget.onCategoryChanged?.call(v);
                      }
                    },
                    decoration: InputDecoration(
                      labelText: 'Correct / Confirm Category',
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

            const SizedBox(height: 8),
            ExpansionTile(
              title: const Text('OCR Text'),
              children: [
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Text(widget.res.text),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
