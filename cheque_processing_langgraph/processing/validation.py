def validate_account_details(account_number: str) -> (bool, str):
    """
    Mock function to validate account details via a banking API.
    """
    print(f"INFO: Validating account details for {account_number} via mock API...")
    # This mock logic considers any account number containing '123' as valid.
    if account_number and "123" in account_number:
        return True, "Account details are valid."
    return False, "Invalid or closed account."

# The trigger_lien_marking function has been removed.